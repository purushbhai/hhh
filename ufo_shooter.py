import math
import random
import sys
from dataclasses import dataclass, field

import numpy as np
import pygame


pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

SAMPLE_RATE = 44100


# --- Config ------------------------------------------------------------------
WIDTH, HEIGHT = 900, 600
FPS = 60

BG_COLOR = (5, 10, 25)
PLAYER_COLOR = (60, 200, 255)
PLAYER_SHIELD_COLOR = (100, 220, 255)
UFO_COLOR_SMALL = (140, 255, 140)
UFO_COLOR_BIG = (255, 120, 180)
BULLET_COLOR = (255, 220, 90)
ENEMY_BULLET_COLOR = (255, 80, 80)
EXPLOSION_COLOR = (255, 140, 80)
BLAST_COLOR = (255, 210, 120)
POWERUP_COLORS = {
    "rapid": (120, 220, 255),
    "spread": (255, 200, 120),
    "shield": (180, 255, 220),
}
HUD_COLOR = (240, 240, 240)

MAX_HEALTH = 3
BLAST_COOLDOWN_MAX = 3.0


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UFO Shooter: Skill Edition")
clock = pygame.time.Clock()

font_small = pygame.font.SysFont("arial", 20)
font_big = pygame.font.SysFont("arial", 40, bold=True)


# --- Audio helpers -----------------------------------------------------------
def make_tone(freq: float, duration: float, volume: float = 0.5) -> pygame.mixer.Sound:
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wave = np.sin(2 * math.pi * freq * t)
    audio = (wave * 32767 * volume).astype(np.int16)
    return pygame.mixer.Sound(buffer=audio)


def make_music_track() -> pygame.mixer.Sound:
    """Generate a short looping synth track."""
    bpm = 96
    beats = 16
    seconds = beats * 60 / bpm
    n_samples = int(SAMPLE_RATE * seconds)
    t = np.linspace(0, seconds, n_samples, endpoint=False)

    # Simple repeating minor chord arpeggio
    scale = [220.0, 261.63, 293.66, 329.63, 349.23, 392.0]  # A minor-ish
    pattern = [0, 2, 4, 5, 4, 2]

    beat_len = seconds / beats
    wave = np.zeros_like(t)
    for i in range(beats):
        f1 = scale[pattern[i % len(pattern)]]
        f2 = scale[(pattern[(i + 2) % len(pattern)])]
        start = int(i * beat_len * SAMPLE_RATE)
        end = int((i + 1) * beat_len * SAMPLE_RATE)
        ti = t[start:end] - t[start]
        chunk = (
            0.6 * np.sin(2 * math.pi * f1 * ti)
            + 0.4 * np.sin(2 * math.pi * f2 * ti * 0.5)
        )
        env = np.linspace(1.0, 0.0, end - start)
        wave[start:end] += chunk * env

    # gentle low-pass by averaging with a slightly shifted copy
    wave = (wave + np.roll(wave, 1)) * 0.5
    wave /= max(1e-6, np.max(np.abs(wave)))

    audio = (wave * 32767 * 0.4).astype(np.int16)
    return pygame.mixer.Sound(buffer=audio)


def make_music_track_alt() -> pygame.mixer.Sound:
    """A second, higher-energy looping track for later difficulty."""
    bpm = 120
    beats = 16
    seconds = beats * 60 / bpm
    n_samples = int(SAMPLE_RATE * seconds)
    t = np.linspace(0, seconds, n_samples, endpoint=False)

    scale = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25]  # C major-ish
    pattern = [0, 2, 4, 5, 4, 2, 1, 3]

    beat_len = seconds / beats
    wave = np.zeros_like(t)
    for i in range(beats):
        f1 = scale[pattern[i % len(pattern)]]
        f2 = scale[(pattern[(i + 3) % len(pattern)])]
        start = int(i * beat_len * SAMPLE_RATE)
        end = int((i + 1) * beat_len * SAMPLE_RATE)
        ti = t[start:end] - t[start]
        base = np.sin(2 * math.pi * f1 * ti)
        high = np.sin(2 * math.pi * f2 * ti * 1.5)
        chunk = 0.6 * base + 0.4 * high
        env = np.concatenate(
            [
                np.linspace(0.0, 1.0, (end - start) // 4, endpoint=False),
                np.linspace(1.0, 0.0, (end - start) - (end - start) // 4),
            ]
        )
        wave[start:end] += chunk * env

    wave = (wave + np.roll(wave, 1) + np.roll(wave, -1)) / 3.0
    wave /= max(1e-6, np.max(np.abs(wave)))

    audio = (wave * 32767 * 0.45).astype(np.int16)
    return pygame.mixer.Sound(buffer=audio)


# --- Entities ----------------------------------------------------------------
@dataclass
class Player:
    x: float
    y: float
    width: int = 50
    height: int = 60
    speed: float = 360.0
    health: int = MAX_HEALTH
    shield_charges: int = 0

    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.width / 2),
            int(self.y - self.height / 2),
            self.width,
            self.height,
        )

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper):
        move_dir = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move_dir -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move_dir += 1

        self.x += move_dir * self.speed * dt
        self.x = max(self.width / 2, min(WIDTH - self.width / 2, self.x))

    def draw(self, surf: pygame.Surface):
        r = self.rect()
        color = PLAYER_SHIELD_COLOR if self.shield_charges > 0 else PLAYER_COLOR
        pygame.draw.rect(surf, color, r, border_radius=8)
        # gun barrel
        barrel = pygame.Rect(r.centerx - 5, r.top - 20, 10, 25)
        pygame.draw.rect(surf, BULLET_COLOR, barrel, border_radius=4)
        # shield ring
        if self.shield_charges > 0:
            pygame.draw.ellipse(
                surf,
                (150, 240, 255),
                r.inflate(20, 10),
                width=2,
            )


@dataclass
class Bullet:
    x: float
    y: float
    vx: float = 0.0
    vy: float = -700.0
    radius: int = 4

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def offscreen(self) -> bool:
        return self.y + self.radius < 0 or self.y - self.radius > HEIGHT

    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )

    def draw(self, surf: pygame.Surface, enemy: bool = False):
        color = ENEMY_BULLET_COLOR if enemy else BULLET_COLOR
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), self.radius)


@dataclass
class UFO:
    x: float
    y: float
    speed: float
    amplitude: float
    phase: float
    width: int
    height: int
    max_health: int
    kind: str  # "small" or "big"
    health: int = field(init=False)
    fire_cooldown: float = field(default_factory=lambda: random.uniform(1.5, 3.0))

    def __post_init__(self):
        self.health = self.max_health

    def update(self, dt: float):
        self.y += self.speed * dt
        self.x += math.sin(pygame.time.get_ticks() / 400 + self.phase) * self.amplitude * dt * 4
        self.fire_cooldown -= dt

    def offscreen(self) -> bool:
        return self.y - self.height / 2 > HEIGHT + 60

    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.width / 2),
            int(self.y - self.height / 2),
            self.width,
            self.height,
        )

    def draw(self, surf: pygame.Surface):
        r = self.rect()
        color = UFO_COLOR_BIG if self.kind == "big" else UFO_COLOR_SMALL
        pygame.draw.ellipse(surf, color, r)
        dome = pygame.Rect(
            r.centerx - r.width // 6,
            r.top - r.height // 3,
            r.width // 3,
            r.height // 2,
        )
        pygame.draw.ellipse(surf, (200, 255, 255), dome)
        for i in range(4 if self.kind == "small" else 6):
            lx = r.left + (i + 0.5) * r.width / (4 if self.kind == "small" else 6)
            pygame.draw.circle(
                surf,
                (255, 255, 180),
                (int(lx), r.bottom - 4),
                4 if self.kind == "small" else 5,
            )

        # health bar
        if self.max_health > 1:
            ratio = self.health / self.max_health
            bar_w = r.width
            bar_h = 5
            bar_x = r.left
            bar_y = r.top - 8
            pygame.draw.rect(surf, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(
                surf,
                (80, 255, 120),
                (bar_x, bar_y, int(bar_w * ratio), bar_h),
            )


@dataclass
class PowerUp:
    x: float
    y: float
    kind: str  # "rapid", "spread", "shield"
    vy: float = 130.0
    size: int = 18

    def update(self, dt: float):
        self.y += self.vy * dt

    def offscreen(self) -> bool:
        return self.y - self.size > HEIGHT + 40

    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.size / 2),
            int(self.y - self.size / 2),
            self.size,
            self.size,
        )

    def draw(self, surf: pygame.Surface):
        color = POWERUP_COLORS.get(self.kind, (255, 255, 255))
        r = self.rect()
        pygame.draw.rect(surf, color, r, border_radius=6)
        letter = {"rapid": "R", "spread": "S", "shield": "H"}.get(self.kind, "?")
        txt = font_small.render(letter, True, (15, 15, 25))
        surf.blit(
            txt,
            (r.centerx - txt.get_width() / 2, r.centery - txt.get_height() / 2),
        )


@dataclass
class BlastWave:
    x: float
    y: float
    lifetime: float = 0.6
    age: float = 0.0

    def update(self, dt: float):
        self.age += dt

    def finished(self) -> bool:
        return self.age >= self.lifetime

    def draw(self, surf: pygame.Surface):
        t = self.age / self.lifetime
        radius = int(80 + 500 * t)
        alpha = max(0, int(220 * (1.0 - t)))
        temp = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            temp,
            (*BLAST_COLOR, alpha),
            (radius, radius),
            radius,
            width=8,
        )
        surf.blit(
            temp,
            (self.x - radius, self.y - radius),
            special_flags=pygame.BLEND_PREMULTIPLIED,
        )


# --- Rendering helpers -------------------------------------------------------
def draw_background(surf: pygame.Surface, time_ms: int, theme_index: int):
    # Different sky palettes as time goes on
    if theme_index == 0:
        base_color = BG_COLOR
        star_layers = [
            (1, 0.02, (40, 40, 70)),
            (2, 0.05, (80, 80, 120)),
            (3, 0.09, (140, 140, 200)),
        ]
    elif theme_index == 1:
        base_color = (10, 5, 25)
        star_layers = [
            (1, 0.03, (70, 40, 80)),
            (2, 0.06, (130, 70, 140)),
            (3, 0.11, (210, 120, 220)),
        ]
    else:
        base_color = (5, 20, 25)
        star_layers = [
            (1, 0.03, (40, 70, 80)),
            (2, 0.07, (80, 150, 170)),
            (3, 0.12, (140, 230, 240)),
        ]

    surf.fill(base_color)
    random.seed(0)
    for layer, speed, color in star_layers:
        for _ in range(90):
            x = random.randint(0, WIDTH)
            base_y = random.randint(0, HEIGHT)
            y = int((base_y + time_ms * speed) % HEIGHT)
            size = layer
            pygame.draw.circle(surf, color, (x, y), size)


def draw_hud(
    surf: pygame.Surface,
    player: Player,
    score: int,
    rapid_timer: float,
    spread_timer: float,
    blast_cooldown: float,
):
    health_text = font_small.render(
        f"Health: {player.health}/{MAX_HEALTH}", True, HUD_COLOR
    )
    score_text = font_small.render(f"Score: {score}", True, HUD_COLOR)
    surf.blit(health_text, (20, 15))
    surf.blit(score_text, (20, 40))

    x = WIDTH - 220
    y = 15
    # Blast status
    if blast_cooldown <= 0:
        t = font_small.render("Blast READY (A)", True, (255, 220, 180))
    else:
        t = font_small.render(f"Blast: {blast_cooldown:0.1f}s", True, (200, 160, 140))
    surf.blit(t, (x, y))
    y += 22

    if rapid_timer > 0:
        t = font_small.render("Rapid fire", True, POWERUP_COLORS["rapid"])
        surf.blit(t, (x, y))
        y += 22
    if spread_timer > 0:
        t = font_small.render("Spread shot", True, POWERUP_COLORS["spread"])
        surf.blit(t, (x, y))
        y += 22
    if player.shield_charges > 0:
        t = font_small.render(
            f"Shield x{player.shield_charges}", True, POWERUP_COLORS["shield"]
        )
        surf.blit(t, (x, y))

    # Blast cooldown bar at bottom center
    bar_width = 220
    bar_height = 12
    bar_x = (WIDTH - bar_width) // 2
    bar_y = HEIGHT - 30

    pygame.draw.rect(
        surf,
        (30, 30, 50),
        (bar_x, bar_y, bar_width, bar_height),
        border_radius=6,
    )

    ratio = 1.0 - min(1.0, max(0.0, blast_cooldown / BLAST_COOLDOWN_MAX))
    if ratio > 0:
        pygame.draw.rect(
            surf,
            BLAST_COLOR,
            (bar_x, bar_y, int(bar_width * ratio), bar_height),
            border_radius=6,
        )

    label = font_small.render("Blast (A)", True, HUD_COLOR)
    surf.blit(
        label,
        (bar_x + bar_width / 2 - label.get_width() / 2, bar_y - label.get_height()),
    )


# --- Game loop ---------------------------------------------------------------
def main():
    # Sounds
    shoot_sound = make_tone(880.0, 0.08, 0.4)
    explosion_sound = make_tone(160.0, 0.35, 0.7)
    hit_sound = make_tone(220.0, 0.25, 0.6)
    pickup_sound = make_tone(1200.0, 0.15, 0.5)
    blast_sound = make_tone(80.0, 0.7, 0.9)

    music_tracks = [make_music_track(), make_music_track_alt()]
    current_music_index = 0
    music = music_tracks[current_music_index]
    music.play(loops=-1)

    player = Player(WIDTH / 2, HEIGHT - 80)
    bullets: list[Bullet] = []
    enemy_bullets: list[Bullet] = []
    ufos: list[UFO] = []
    powerups: list[PowerUp] = []
    blast_effects: list[BlastWave] = []

    score = 0
    elapsed_time = 0.0
    spawn_timer = 0.0
    spawn_interval = 1.0
    shoot_cooldown_base = 0.25
    shoot_cooldown_rapid = 0.09
    shoot_timer = 0.0

    rapid_fire_timer = 0.0
    spread_shot_timer = 0.0
    blast_cooldown = 0.0

    running = True
    game_over = False

    while running:
        dt = clock.tick(FPS) / 1000.0
        time_ms = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if game_over and event.key == pygame.K_r:
                    music.stop()
                    return main()
                # Screen-clearing blast on 'A' key when not on cooldown
                if (
                    not game_over
                    and event.key == pygame.K_a
                    and blast_cooldown <= 0.0
                ):
                    if ufos or enemy_bullets:
                        blast_sound.play()
                        blast_effects.append(
                            BlastWave(player.x, player.y - player.height / 2 - 10)
                        )
                    for u in ufos:
                        score += 3 if u.kind == "big" else 1
                    ufos.clear()
                    enemy_bullets.clear()
                    blast_cooldown = 3.0

        keys = pygame.key.get_pressed()

        if not game_over:
            elapsed_time += dt
            # Update timers
            shoot_timer -= dt
            rapid_fire_timer = max(0.0, rapid_fire_timer - dt)
            spread_shot_timer = max(0.0, spread_shot_timer - dt)
            blast_cooldown = max(0.0, blast_cooldown - dt)

            # Switch to second music track when difficulty ramps up
            if elapsed_time > 60.0 and current_music_index == 0:
                music.stop()
                current_music_index = 1
                music = music_tracks[current_music_index]
                music.play(loops=-1)

            # Update player
            player.update(dt, keys)

            # Shooting (skill-modified)
            fire_delay = (
                shoot_cooldown_rapid if rapid_fire_timer > 0 else shoot_cooldown_base
            )
            if (keys[pygame.K_SPACE] or keys[pygame.K_UP]) and shoot_timer <= 0:
                # center bullet
                bullet_list = [Bullet(player.x, player.y - player.height / 2 - 20)]
                # spread shot skill
                if spread_shot_timer > 0:
                    bullet_list.append(
                        Bullet(player.x, player.y - player.height / 2 - 20, vx=-180.0)
                    )
                    bullet_list.append(
                        Bullet(player.x, player.y - player.height / 2 - 20, vx=180.0)
                    )
                bullets.extend(bullet_list)
                shoot_timer = fire_delay
                shoot_sound.play()

            # Difficulty scaling over time
            difficulty_factor = 1.0 + elapsed_time / 45.0
            spawn_interval_target = max(0.5, 1.6 / difficulty_factor)
            max_ufos = min(4 + int(elapsed_time / 30.0) * 2, 14)

            # Spawn UFOs (mix of small and big, limited by max_ufos)
            spawn_timer -= dt
            if spawn_timer <= 0.0 and len(ufos) < max_ufos:
                x = random.randint(80, WIDTH - 80)
                y = -60
                is_big = random.random() < min(0.25 + elapsed_time / 120.0, 0.55)

                if is_big:
                    speed = random.uniform(70, 110)
                    width, height = 100, 50
                    health = 4
                    amplitude = random.uniform(20, 40)
                else:
                    speed = random.uniform(100, 160)
                    width, height = 60, 30
                    health = 1
                    amplitude = random.uniform(20, 50)

                phase = random.uniform(0, math.pi * 2)
                ufos.append(
                    UFO(
                        x=x,
                        y=y,
                        speed=speed,
                        amplitude=amplitude,
                        phase=phase,
                        width=width,
                        height=height,
                        max_health=health,
                        kind="big" if is_big else "small",
                    )
                )

                spawn_interval = spawn_interval_target
                spawn_timer = spawn_interval

            # Update bullets
            for b in bullets:
                b.update(dt)
            bullets = [b for b in bullets if not b.offscreen()]

            for eb in enemy_bullets:
                eb.update(dt)
            enemy_bullets = [eb for eb in enemy_bullets if not eb.offscreen()]

            # Update UFOs and enemy fire
            for u in ufos:
                u.update(dt)
                if u.kind == "big" and u.fire_cooldown <= 0 and u.y > 40:
                    enemy_bullets.append(
                        Bullet(u.x, u.y + u.height / 2, vy=260.0, radius=5)
                    )
                    u.fire_cooldown = random.uniform(1.5, 3.0)

            # Bullet vs UFO collisions
            for b in bullets[:]:
                b_rect = b.rect()
                for u in ufos[:]:
                    if b_rect.colliderect(u.rect()):
                        bullets.remove(b)
                        u.health -= 1
                        if u.health <= 0:
                            ufos.remove(u)
                            score += 3 if u.kind == "big" else 1
                            explosion_sound.play()

                            # power-up drop
                            if random.random() < (0.35 if u.kind == "big" else 0.18):
                                kind = random.choice(["rapid", "spread", "shield"])
                                powerups.append(PowerUp(u.x, u.y, kind=kind))
                        else:
                            hit_sound.play()
                        break

            # UFO vs player and bottom
            for u in ufos[:]:
                if u.rect().colliderect(player.rect()) or u.y > HEIGHT - 30:
                    ufos.remove(u)
                    explosion_sound.play()
                    if player.shield_charges > 0:
                        player.shield_charges -= 1
                    else:
                        player.health -= 1
                    if player.health <= 0:
                        game_over = True

            # Enemy bullets vs player
            for eb in enemy_bullets[:]:
                if eb.rect().colliderect(player.rect()):
                    enemy_bullets.remove(eb)
                    explosion_sound.play()
                    if player.shield_charges > 0:
                        player.shield_charges -= 1
                    else:
                        player.health -= 1
                    if player.health <= 0:
                        game_over = True

            # Power-ups
            for p in powerups:
                p.update(dt)
            powerups = [p for p in powerups if not p.offscreen()]

            for p in powerups[:]:
                if p.rect().colliderect(player.rect()):
                    powerups.remove(p)
                    pickup_sound.play()
                    if p.kind == "rapid":
                        rapid_fire_timer = 7.0
                    elif p.kind == "spread":
                        spread_shot_timer = 7.0
                    elif p.kind == "shield":
                        player.shield_charges = min(3, player.shield_charges + 1)

        # --- Drawing ---
        offset_x = int((player.x - WIDTH / 2) * 0.06)
        camera_surf = pygame.Surface((WIDTH, HEIGHT))

        # Theme/index changes background colors as time increases
        theme_index = int(elapsed_time // 45.0) % 3
        draw_background(camera_surf, time_ms, theme_index)

        for u in ufos:
            u.draw(camera_surf)
        for p in powerups:
            p.draw(camera_surf)
        for b in bullets:
            b.draw(camera_surf)
        for eb in enemy_bullets:
            eb.draw(camera_surf, enemy=True)

        # Blast wave animations
        for bw in blast_effects:
            bw.update(dt)
            bw.draw(camera_surf)
        blast_effects[:] = [bw for bw in blast_effects if not bw.finished()]

        player.draw(camera_surf)

        screen.blit(camera_surf, (-offset_x, 0))

        draw_hud(screen, player, score, rapid_fire_timer, spread_shot_timer, blast_cooldown)

        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            screen.blit(overlay, (0, 0))

            msg = "You were destroyed!"
            msg2 = "Press R to restart or ESC to quit"
            t1 = font_big.render(msg, True, (255, 230, 230))
            t2 = font_small.render(msg2, True, (230, 230, 230))
            screen.blit(
                t1,
                (WIDTH / 2 - t1.get_width() / 2, HEIGHT / 2 - 50),
            )
            screen.blit(
                t2,
                (WIDTH / 2 - t2.get_width() / 2, HEIGHT / 2 + 0),
            )

        # Lightweight "link" text you can copy/share
        link_text = font_small.render(
            "Game link: https://ufo-shooter.local", True, (150, 150, 190)
        )
        screen.blit(link_text, (20, HEIGHT - 22))

        pygame.display.flip()

    music.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

