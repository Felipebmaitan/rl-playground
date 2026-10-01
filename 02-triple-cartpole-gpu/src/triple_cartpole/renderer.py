from __future__ import annotations

import math

import pygame

from .config import PhysicsConfig


class TripleCartPoleRenderer:
    def __init__(self, physics: PhysicsConfig, width: int = 1100, height: int = 700) -> None:
        pygame.init()
        pygame.display.set_caption("GPU Triple CartPole")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.physics = physics
        self.width = width
        self.height = height
        self.running = True

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
        return self.running

    def draw(self, state: list[float], reward: float, episode_step: int, fps: int = 60) -> None:
        self.screen.fill((18, 20, 28))
        track_y = int(self.height * 0.70)
        margin = 100
        scale_x = (self.width - 2 * margin) / (2.0 * self.physics.track_limit)
        cart_x = int(self.width / 2 + state[0] * scale_x)
        pygame.draw.line(self.screen, (130, 140, 155), (margin, track_y), (self.width - margin, track_y), 5)
        for tick in range(-2, 3):
            tick_x = int(self.width / 2 + tick * scale_x)
            pygame.draw.line(self.screen, (80, 88, 100), (tick_x, track_y - 8), (tick_x, track_y + 8), 2)

        cart_rect = pygame.Rect(cart_x - 58, track_y - 38, 116, 38)
        pygame.draw.rect(self.screen, (70, 160, 235), cart_rect, border_radius=7)
        pygame.draw.circle(self.screen, (25, 28, 36), (cart_x - 37, track_y + 3), 14)
        pygame.draw.circle(self.screen, (25, 28, 36), (cart_x + 37, track_y + 3), 14)

        pixels_per_meter = 235
        joint = (float(cart_x), float(track_y - 38))
        colors = ((244, 112, 112), (252, 190, 75), (105, 219, 160))
        for angle, length, color in zip(state[1:4], self.physics.link_lengths, colors, strict=True):
            next_joint = (
                joint[0] + pixels_per_meter * length * math.sin(angle),
                joint[1] - pixels_per_meter * length * math.cos(angle),
            )
            pygame.draw.line(self.screen, color, joint, next_joint, 12)
            pygame.draw.circle(self.screen, (235, 240, 248), (round(joint[0]), round(joint[1])), 10)
            joint = next_joint
        pygame.draw.circle(self.screen, (235, 240, 248), (round(joint[0]), round(joint[1])), 9)

        status = self.font.render(
            f"reward {reward:+.3f}   step {episode_step}   ESC to close", True, (225, 230, 240)
        )
        self.screen.blit(status, (24, 24))
        pygame.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()

