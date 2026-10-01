from __future__ import annotations

import warp as wp


NUM_LINKS = 5
vec5 = wp.types.vector(length=5, dtype=float)
vec6 = wp.types.vector(length=6, dtype=float)
mat66 = wp.types.matrix(shape=(6, 6), dtype=float)


@wp.struct
class EnvParams:
    cart_mass: float
    mass1: float
    mass2: float
    mass3: float
    mass4: float
    mass5: float
    length1: float
    length2: float
    length3: float
    length4: float
    length5: float
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
def solve_spd_6x6(matrix: mat66, rhs: vec6) -> vec6:
    """Small register-resident LDL^T solve for the articulated mass matrix."""

    lower = mat66()
    diagonal = vec6()
    for row in range(6):
        for column in range(row):
            value = matrix[row, column]
            for previous in range(column):
                value = value - lower[row, previous] * lower[column, previous] * diagonal[previous]
            lower[row, column] = value / diagonal[column]
        diagonal_value = matrix[row, row]
        for previous in range(row):
            diagonal_value = diagonal_value - lower[row, previous] * lower[row, previous] * diagonal[previous]
        diagonal[row] = diagonal_value
        lower[row, row] = 1.0

    forward = vec6()
    scaled = vec6()
    for row in range(6):
        value = rhs[row]
        for column in range(row):
            value = value - lower[row, column] * forward[column]
        forward[row] = value
        scaled[row] = value / diagonal[row]

    solution = vec6()
    for reverse_index in range(6):
        row = 5 - reverse_index
        value = scaled[row]
        for column in range(row + 1, 6):
            value = value - lower[column, row] * solution[column]
        solution[row] = value
    return solution


@wp.func
def write_observation(
    env_id: int,
    track_limit: float,
    state: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    observation: wp.array2d(dtype=float),
):
    observation[env_id, 0] = state[env_id, 0] / track_limit
    observation[env_id, 1] = state[env_id, 6] * 0.2
    for link in range(NUM_LINKS):
        angle = state[env_id, 1 + link]
        observation[env_id, 2 + 2 * link] = wp.sin(angle)
        observation[env_id, 3 + 2 * link] = wp.cos(angle)
        observation[env_id, 12 + link] = state[env_id, 7 + link] * 0.1
    observation[env_id, 17] = action[env_id, 0]


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

    state[env_id, 0] = wp.randf(
        random_state, -params.reset_position_noise, params.reset_position_noise
    )
    for link in range(NUM_LINKS):
        state[env_id, 1 + link] = angle_center + wp.randf(
            random_state, -params.reset_angle_noise, params.reset_angle_noise
        )
    state[env_id, 6] = wp.randf(
        random_state, -params.reset_velocity_noise, params.reset_velocity_noise
    )
    for link in range(NUM_LINKS):
        state[env_id, 7 + link] = wp.randf(
            random_state, -params.reset_velocity_noise, params.reset_velocity_noise
        )
    action[env_id, 0] = 0.0
    episode_step[env_id] = 0
    episode_return[env_id] = 0.0
    write_observation(env_id, params.track_limit, state, action, observation)


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
    rollout_return_sum: wp.array(dtype=float),
    rollout_episode_count: wp.array(dtype=wp.int32),
):
    env_id = wp.tid()
    masses = vec5(params.mass1, params.mass2, params.mass3, params.mass4, params.mass5)
    lengths = vec5(
        params.length1, params.length2, params.length3, params.length4, params.length5
    )

    # h[j] is the first moment of every body moved by joint j. The off-diagonal
    # constant H[j,k] simplifies to length[min(j,k)] * h[max(j,k)].
    h = vec5(
        params.mass1 * 0.5 * params.length1
        + params.length1 * (params.mass2 + params.mass3 + params.mass4 + params.mass5),
        params.mass2 * 0.5 * params.length2
        + params.length2 * (params.mass3 + params.mass4 + params.mass5),
        params.mass3 * 0.5 * params.length3
        + params.length3 * (params.mass4 + params.mass5),
        params.mass4 * 0.5 * params.length4 + params.length4 * params.mass5,
        params.mass5 * 0.5 * params.length5,
    )
    trailing_mass = vec5(
        params.mass2 + params.mass3 + params.mass4 + params.mass5,
        params.mass3 + params.mass4 + params.mass5,
        params.mass4 + params.mass5,
        params.mass5,
        0.0,
    )
    angular_diagonal = vec5()
    total_mass = params.cart_mass
    for link in range(NUM_LINKS):
        length = lengths[link]
        mass = masses[link]
        center = 0.5 * length
        inertia = mass * length * length / 12.0
        angular_diagonal[link] = (
            mass * center * center + trailing_mass[link] * length * length + inertia
        )
        total_mass = total_mass + mass

    normalized_action = wp.clamp(action[env_id, 0], -1.0, 1.0)
    action[env_id, 0] = normalized_action
    force = normalized_action * params.max_force

    for substep in range(params.substeps):
        angles = vec5()
        angular_velocity = vec5()
        for link in range(NUM_LINKS):
            angles[link] = state[env_id, 1 + link]
            angular_velocity[link] = state[env_id, 7 + link]
        cart_velocity = state[env_id, 6]

        mass_matrix = mat66()
        mass_matrix[0, 0] = total_mass
        for row in range(NUM_LINKS):
            cart_coupling = h[row] * wp.cos(angles[row])
            mass_matrix[0, row + 1] = cart_coupling
            mass_matrix[row + 1, 0] = cart_coupling
            for column in range(NUM_LINKS):
                constant = angular_diagonal[row]
                if row < column:
                    constant = lengths[row] * h[column]
                elif row > column:
                    constant = lengths[column] * h[row]
                mass_matrix[row + 1, column + 1] = constant * wp.cos(
                    angles[row] - angles[column]
                )

        rhs = vec6()
        cart_bias = params.cart_damping * cart_velocity
        for link in range(NUM_LINKS):
            cart_bias = cart_bias - h[link] * wp.sin(angles[link]) * angular_velocity[link] * angular_velocity[link]
        rhs[0] = force - cart_bias
        for row in range(NUM_LINKS):
            angular_bias = (
                -params.gravity * h[row] * wp.sin(angles[row])
                + params.joint_damping * angular_velocity[row]
            )
            for column in range(NUM_LINKS):
                if row != column:
                    constant = angular_diagonal[row]
                    if row < column:
                        constant = lengths[row] * h[column]
                    else:
                        constant = lengths[column] * h[row]
                    angular_bias = angular_bias + constant * wp.sin(
                        angles[row] - angles[column]
                    ) * angular_velocity[column] * angular_velocity[column]
            rhs[row + 1] = -angular_bias

        acceleration = solve_spd_6x6(mass_matrix, rhs)
        cart_velocity = cart_velocity + acceleration[0] * params.physics_dt
        state[env_id, 6] = cart_velocity
        state[env_id, 0] = state[env_id, 0] + cart_velocity * params.physics_dt
        for link in range(NUM_LINKS):
            omega = angular_velocity[link] + acceleration[link + 1] * params.physics_dt
            angle = angles[link] + omega * params.physics_dt
            state[env_id, 7 + link] = omega
            state[env_id, 1 + link] = wp.atan2(wp.sin(angle), wp.cos(angle))

    episode_step[env_id] = episode_step[env_id] + 1
    x = state[env_id, 0]
    cart_velocity = state[env_id, 6]
    failed = False
    if params.terminate_at_track_limit == 1 and wp.abs(x) >= params.track_limit:
        failed = True
    if not wp.isfinite(x) or not wp.isfinite(cart_velocity):
        failed = True
    if wp.abs(cart_velocity) > 100.0:
        failed = True
    for link in range(NUM_LINKS):
        if (
            not wp.isfinite(state[env_id, 1 + link])
            or not wp.isfinite(state[env_id, 7 + link])
            or wp.abs(state[env_id, 7 + link]) > 200.0
        ):
            failed = True

    shaped_reward = params.terminal_penalty
    if not failed:
        upright_sum = 0.0
        upright_product = 1.0
        alignment_cosine_sum = 0.0
        angular_speed_squared = 0.0
        precision = True
        for link in range(NUM_LINKS):
            angle = state[env_id, 1 + link]
            omega = state[env_id, 7 + link]
            cosine = wp.cos(angle)
            upright = 0.5 * (cosine + 1.0)
            upright_sum = upright_sum + upright
            upright_product = upright_product * upright
            angular_speed_squared = angular_speed_squared + omega * omega
            if cosine <= params.precision_cosine or wp.abs(omega) >= params.precision_angular_velocity:
                precision = False
            if link < NUM_LINKS - 1:
                alignment_cosine_sum = alignment_cosine_sum + wp.cos(
                    angle - state[env_id, 2 + link]
                )

        mean_upright = upright_sum / 5.0
        all_upright = wp.pow(wp.max(upright_product, 0.0), 0.2)
        alignment = 0.5 + 0.5 * alignment_cosine_sum / 4.0
        stillness = wp.exp(-params.stillness_coefficient * angular_speed_squared)
        stable_straight = all_upright * alignment * stillness
        if (
            wp.abs(cart_velocity) >= params.precision_cart_velocity
            or wp.abs(x) >= params.precision_position_fraction * params.track_limit
        ):
            precision = False
        precision_bonus = 0.0
        if precision:
            precision_bonus = 1.0

        shaped_reward = (
            params.mean_upright_weight * mean_upright
            + params.all_upright_weight * all_upright
            + params.stable_straight_weight * stable_straight
            + params.precision_bonus_weight * precision_bonus
            - params.position_cost * (x / params.track_limit) * (x / params.track_limit)
            - params.velocity_cost
            * (cart_velocity * cart_velocity + 0.1 * angular_speed_squared)
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
        wp.atomic_add(rollout_return_sum, 0, episode_return[env_id])
        wp.atomic_add(rollout_episode_count, 0, 1)
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
        write_observation(env_id, params.track_limit, state, action, observation)
