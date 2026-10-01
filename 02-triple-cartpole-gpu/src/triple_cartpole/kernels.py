from __future__ import annotations

import warp as wp


@wp.struct
class EnvParams:
    cart_mass: float
    mass1: float
    mass2: float
    mass3: float
    length1: float
    length2: float
    length3: float
    gravity: float
    cart_damping: float
    joint_damping: float
    max_force: float
    physics_dt: float
    substeps: int
    track_limit: float
    max_episode_steps: int
    task_id: int
    reset_position_noise: float
    reset_angle_noise: float
    reset_velocity_noise: float
    mean_upright_weight: float
    all_upright_weight: float
    stable_straight_weight: float
    precision_bonus_weight: float
    stillness_coefficient: float
    precision_cosine: float
    precision_angular_velocity: float
    precision_cart_velocity: float
    precision_position_fraction: float
    position_cost: float
    velocity_cost: float
    action_cost: float
    terminal_penalty: float
    terminate_at_track_limit: int
    seed: int


@wp.func
def solve_spd_4x4(matrix: wp.mat44, rhs: wp.vec4) -> wp.vec4:
    """Register-only LDL^T solve for the symmetric positive-definite mass matrix."""
    d0 = matrix[0, 0]
    l10 = matrix[1, 0] / d0
    l20 = matrix[2, 0] / d0
    l30 = matrix[3, 0] / d0

    d1 = matrix[1, 1] - l10 * l10 * d0
    l21 = (matrix[2, 1] - l20 * l10 * d0) / d1
    l31 = (matrix[3, 1] - l30 * l10 * d0) / d1

    d2 = matrix[2, 2] - l20 * l20 * d0 - l21 * l21 * d1
    l32 = (matrix[3, 2] - l30 * l20 * d0 - l31 * l21 * d1) / d2
    d3 = matrix[3, 3] - l30 * l30 * d0 - l31 * l31 * d1 - l32 * l32 * d2

    y0 = rhs[0]
    y1 = rhs[1] - l10 * y0
    y2 = rhs[2] - l20 * y0 - l21 * y1
    y3 = rhs[3] - l30 * y0 - l31 * y1 - l32 * y2

    x3 = y3 / d3
    x2 = y2 / d2 - l32 * x3
    x1 = y1 / d1 - l21 * x2 - l31 * x3
    x0 = y0 / d0 - l10 * x1 - l20 * x2 - l30 * x3
    return wp.vec4(x0, x1, x2, x3)


@wp.func
def write_observation(
    env_id: int,
    track_limit: float,
    max_force: float,
    state: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    observation: wp.array2d(dtype=float),
):
    theta1 = state[env_id, 1]
    theta2 = state[env_id, 2]
    theta3 = state[env_id, 3]
    observation[env_id, 0] = state[env_id, 0] / track_limit
    observation[env_id, 1] = state[env_id, 4] * 0.2
    observation[env_id, 2] = wp.sin(theta1)
    observation[env_id, 3] = wp.cos(theta1)
    observation[env_id, 4] = wp.sin(theta2)
    observation[env_id, 5] = wp.cos(theta2)
    observation[env_id, 6] = wp.sin(theta3)
    observation[env_id, 7] = wp.cos(theta3)
    observation[env_id, 8] = state[env_id, 5] * 0.1
    observation[env_id, 9] = state[env_id, 6] * 0.1
    observation[env_id, 10] = state[env_id, 7] * 0.1
    observation[env_id, 11] = action[env_id, 0]


@wp.func
def reset_one(
    env_id: int,
    episode: int,
    params: EnvParams,
    state: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    observation: wp.array2d(dtype=float),
    episode_step: wp.array(dtype=wp.int32),
    episode_return: wp.array(dtype=float),
):
    random_state = wp.rand_init(params.seed, env_id + episode * 104729)
    angle_center = 0.0
    if params.task_id == 1:
        angle_center = wp.pi

    state[env_id, 0] = wp.randf(random_state, -params.reset_position_noise, params.reset_position_noise)
    state[env_id, 1] = angle_center + wp.randf(random_state, -params.reset_angle_noise, params.reset_angle_noise)
    state[env_id, 2] = angle_center + wp.randf(random_state, -params.reset_angle_noise, params.reset_angle_noise)
    state[env_id, 3] = angle_center + wp.randf(random_state, -params.reset_angle_noise, params.reset_angle_noise)
    state[env_id, 4] = wp.randf(random_state, -params.reset_velocity_noise, params.reset_velocity_noise)
    state[env_id, 5] = wp.randf(random_state, -params.reset_velocity_noise, params.reset_velocity_noise)
    state[env_id, 6] = wp.randf(random_state, -params.reset_velocity_noise, params.reset_velocity_noise)
    state[env_id, 7] = wp.randf(random_state, -params.reset_velocity_noise, params.reset_velocity_noise)
    action[env_id, 0] = 0.0
    episode_step[env_id] = 0
    episode_return[env_id] = 0.0
    write_observation(env_id, params.track_limit, params.max_force, state, action, observation)


@wp.kernel
def reset_kernel(
    params: EnvParams,
    state: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    observation: wp.array2d(dtype=float),
    reward: wp.array(dtype=float),
    done: wp.array(dtype=wp.uint8),
    episode_step: wp.array(dtype=wp.int32),
    episode_count: wp.array(dtype=wp.int32),
    episode_return: wp.array(dtype=float),
):
    env_id = wp.tid()
    episode_count[env_id] = 0
    reward[env_id] = 0.0
    done[env_id] = wp.uint8(0)
    reset_one(env_id, 0, params, state, action, observation, episode_step, episode_return)


@wp.kernel
def step_kernel(
    params: EnvParams,
    state: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    observation: wp.array2d(dtype=float),
    reward: wp.array(dtype=float),
    done: wp.array(dtype=wp.uint8),
    episode_step: wp.array(dtype=wp.int32),
    episode_count: wp.array(dtype=wp.int32),
    episode_return: wp.array(dtype=float),
    completed_return: wp.array(dtype=float),
    completed_length: wp.array(dtype=wp.int32),
):
    env_id = wp.tid()

    m1 = params.mass1
    m2 = params.mass2
    m3 = params.mass3
    l1 = params.length1
    l2 = params.length2
    l3 = params.length3
    c1 = 0.5 * l1
    c2 = 0.5 * l2
    c3 = 0.5 * l3
    inertia1 = m1 * l1 * l1 / 12.0
    inertia2 = m2 * l2 * l2 / 12.0
    inertia3 = m3 * l3 * l3 / 12.0

    h1 = m1 * c1 + (m2 + m3) * l1
    h2 = m2 * c2 + m3 * l2
    h3 = m3 * c3
    angular11 = m1 * c1 * c1 + (m2 + m3) * l1 * l1 + inertia1
    angular22 = m2 * c2 * c2 + m3 * l2 * l2 + inertia2
    angular33 = m3 * c3 * c3 + inertia3
    angular12 = m2 * l1 * c2 + m3 * l1 * l2
    angular13 = m3 * l1 * c3
    angular23 = m3 * l2 * c3
    total_mass = params.cart_mass + m1 + m2 + m3

    normalized_action = wp.clamp(action[env_id, 0], -1.0, 1.0)
    action[env_id, 0] = normalized_action
    force = normalized_action * params.max_force

    for substep in range(params.substeps):
        x_velocity = state[env_id, 4]
        theta1 = state[env_id, 1]
        theta2 = state[env_id, 2]
        theta3 = state[env_id, 3]
        omega1 = state[env_id, 5]
        omega2 = state[env_id, 6]
        omega3 = state[env_id, 7]

        mass_matrix = wp.mat44(
            total_mass, h1 * wp.cos(theta1), h2 * wp.cos(theta2), h3 * wp.cos(theta3),
            h1 * wp.cos(theta1), angular11, angular12 * wp.cos(theta1 - theta2), angular13 * wp.cos(theta1 - theta3),
            h2 * wp.cos(theta2), angular12 * wp.cos(theta2 - theta1), angular22, angular23 * wp.cos(theta2 - theta3),
            h3 * wp.cos(theta3), angular13 * wp.cos(theta3 - theta1), angular23 * wp.cos(theta3 - theta2), angular33,
        )

        bias0 = (
            -h1 * wp.sin(theta1) * omega1 * omega1
            -h2 * wp.sin(theta2) * omega2 * omega2
            -h3 * wp.sin(theta3) * omega3 * omega3
            + params.cart_damping * x_velocity
        )
        bias1 = (
            angular12 * wp.sin(theta1 - theta2) * omega2 * omega2
            + angular13 * wp.sin(theta1 - theta3) * omega3 * omega3
            - params.gravity * h1 * wp.sin(theta1)
            + params.joint_damping * omega1
        )
        bias2 = (
            angular12 * wp.sin(theta2 - theta1) * omega1 * omega1
            + angular23 * wp.sin(theta2 - theta3) * omega3 * omega3
            - params.gravity * h2 * wp.sin(theta2)
            + params.joint_damping * omega2
        )
        bias3 = (
            angular13 * wp.sin(theta3 - theta1) * omega1 * omega1
            + angular23 * wp.sin(theta3 - theta2) * omega2 * omega2
            - params.gravity * h3 * wp.sin(theta3)
            + params.joint_damping * omega3
        )
        acceleration = solve_spd_4x4(
            mass_matrix, wp.vec4(force - bias0, -bias1, -bias2, -bias3)
        )

        x_velocity = x_velocity + acceleration[0] * params.physics_dt
        omega1 = omega1 + acceleration[1] * params.physics_dt
        omega2 = omega2 + acceleration[2] * params.physics_dt
        omega3 = omega3 + acceleration[3] * params.physics_dt
        state[env_id, 4] = x_velocity
        state[env_id, 5] = omega1
        state[env_id, 6] = omega2
        state[env_id, 7] = omega3
        state[env_id, 0] = state[env_id, 0] + x_velocity * params.physics_dt
        state[env_id, 1] = wp.atan2(wp.sin(theta1 + omega1 * params.physics_dt), wp.cos(theta1 + omega1 * params.physics_dt))
        state[env_id, 2] = wp.atan2(wp.sin(theta2 + omega2 * params.physics_dt), wp.cos(theta2 + omega2 * params.physics_dt))
        state[env_id, 3] = wp.atan2(wp.sin(theta3 + omega3 * params.physics_dt), wp.cos(theta3 + omega3 * params.physics_dt))

    episode_step[env_id] = episode_step[env_id] + 1
    theta1 = state[env_id, 1]
    theta2 = state[env_id, 2]
    theta3 = state[env_id, 3]
    x = state[env_id, 0]
    x_velocity = state[env_id, 4]
    omega1 = state[env_id, 5]
    omega2 = state[env_id, 6]
    omega3 = state[env_id, 7]

    failed = False
    if params.terminate_at_track_limit == 1 and wp.abs(x) >= params.track_limit:
        failed = True
    if (
        not wp.isfinite(x)
        or not wp.isfinite(theta1)
        or not wp.isfinite(theta2)
        or not wp.isfinite(theta3)
        or not wp.isfinite(x_velocity)
        or not wp.isfinite(omega1)
        or not wp.isfinite(omega2)
        or not wp.isfinite(omega3)
    ):
        failed = True
    # A finite but explosive state is also a numerical failure. Reset before it
    # can create an unbounded reward/advantage and corrupt PPO.
    if wp.abs(x_velocity) > 100.0 or wp.abs(omega1) > 200.0 or wp.abs(omega2) > 200.0 or wp.abs(omega3) > 200.0:
        failed = True

    shaped_reward = params.terminal_penalty
    if not failed:
        cosine1 = wp.cos(theta1)
        cosine2 = wp.cos(theta2)
        cosine3 = wp.cos(theta3)

        # Each link contributes equally. The geometric term collapses toward
        # zero when even one link is down, unlike the old potential-height
        # reward that gave the first link almost 70% of all available points.
        upright1 = 0.5 * (cosine1 + 1.0)
        upright2 = 0.5 * (cosine2 + 1.0)
        upright3 = 0.5 * (cosine3 + 1.0)
        mean_upright = (upright1 + upright2 + upright3) / 3.0
        all_upright = wp.pow(
            wp.max(upright1 * upright2 * upright3, 0.0), 1.0 / 3.0
        )

        # Alignment alone also rewards a straight chain hanging downward, so
        # all_upright gates it. Stillness distinguishes controlled balance from
        # a high-speed pass through the vertical pose.
        alignment = 0.25 * (
            2.0 + wp.cos(theta1 - theta2) + wp.cos(theta2 - theta3)
        )
        angular_speed_squared = omega1 * omega1 + omega2 * omega2 + omega3 * omega3
        stillness = wp.exp(-params.stillness_coefficient * angular_speed_squared)
        stable_straight = all_upright * alignment * stillness

        precision_bonus = 0.0
        if (
            cosine1 > params.precision_cosine
            and cosine2 > params.precision_cosine
            and cosine3 > params.precision_cosine
            and wp.abs(omega1) < params.precision_angular_velocity
            and wp.abs(omega2) < params.precision_angular_velocity
            and wp.abs(omega3) < params.precision_angular_velocity
            and wp.abs(x_velocity) < params.precision_cart_velocity
            and wp.abs(x) < params.precision_position_fraction * params.track_limit
        ):
            precision_bonus = 1.0

        shaped_reward = (
            params.mean_upright_weight * mean_upright
            + params.all_upright_weight * all_upright
            + params.stable_straight_weight * stable_straight
            + params.precision_bonus_weight * precision_bonus
            - params.position_cost * (x / params.track_limit) * (x / params.track_limit)
            - params.velocity_cost * (x_velocity * x_velocity + 0.1 * angular_speed_squared)
            - params.action_cost * normalized_action * normalized_action
        )
        shaped_reward = wp.clamp(shaped_reward, params.terminal_penalty, 1.0)

    reward[env_id] = shaped_reward
    episode_return[env_id] = episode_return[env_id] + shaped_reward
    terminal = episode_step[env_id] >= params.max_episode_steps or failed

    if terminal:
        done[env_id] = wp.uint8(1)
        completed_return[env_id] = episode_return[env_id]
        completed_length[env_id] = episode_step[env_id]
        episode_count[env_id] = episode_count[env_id] + 1
        reset_one(
            env_id,
            episode_count[env_id],
            params,
            state,
            action,
            observation,
            episode_step,
            episode_return,
        )
    else:
        done[env_id] = wp.uint8(0)
        completed_return[env_id] = 0.0
        completed_length[env_id] = 0
        write_observation(env_id, params.track_limit, params.max_force, state, action, observation)
