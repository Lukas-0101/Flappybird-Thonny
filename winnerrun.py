
"""
Flappy Bird - Play with the best trained NEAT bird.
Loads winner.pkl and plays with only one bird.
No training is performed in this file.
"""

import pygame
import random
import os
import neat
import pickle

pygame.init()
pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730

STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Best Bird")

# Find image files relative to this Python file
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(LOCAL_DIR, "imgs")

pipe_img = pygame.transform.scale2x(
    pygame.image.load(
        os.path.join(IMG_DIR, "pipe.png")
    ).convert_alpha()
)

bg_img = pygame.transform.scale(
    pygame.image.load(
        os.path.join(IMG_DIR, "bg.png")
    ).convert_alpha(),
    (600, 900)
)

bird_images = [
    pygame.transform.scale2x(
        pygame.image.load(
            os.path.join(IMG_DIR, "bird" + str(x) + ".png")
        ).convert_alpha()
    )
    for x in range(1, 4)
]

base_img = pygame.transform.scale2x(
    pygame.image.load(
        os.path.join(IMG_DIR, "base.png")
    ).convert_alpha()
)


class Bird:
    """The single bird controlled by the trained neural network."""

    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        displacement = (
            self.vel * self.tick_count
            + 0.5 * 3 * self.tick_count ** 2
        )

        if displacement >= 16:
            displacement = 16

        if displacement < 0:
            displacement -= 2

        self.y += displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        else:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        blitRotateCenter(
            win, self.img, (self.x, self.y), self.tilt
        )

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    """Top and bottom pipe."""

    GAP = 160
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(
            pipe_img, False, True
        )
        self.PIPE_BOTTOM = pipe_img

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (
            self.x - bird.x,
            self.top - round(bird.y)
        )
        bottom_offset = (
            self.x - bird.x,
            self.bottom - round(bird.y)
        )

        top_point = bird_mask.overlap(top_mask, top_offset)
        bottom_point = bird_mask.overlap(
            bottom_mask, bottom_offset
        )

        return top_point is not None or bottom_point is not None


class Base:
    """Scrolling floor."""

    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(
        center=image.get_rect(topleft=topleft).center
    )
    surf.blit(rotated_image, new_rect.topleft)


def draw_window(win, bird, pipes, base, score):
    win.blit(bg_img, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)

    if DRAW_LINES and pipes:
        next_pipe = pipes[0]

        if next_pipe.x + next_pipe.PIPE_TOP.get_width() < bird.x:
            if len(pipes) > 1:
                next_pipe = pipes[1]

        pygame.draw.line(
            win, (255, 0, 0),
            (
                bird.x + bird.img.get_width() / 2,
                bird.y + bird.img.get_height() / 2
            ),
            (
                next_pipe.x + next_pipe.PIPE_TOP.get_width() / 2,
                next_pipe.height
            ),
            3
        )

    bird.draw(win)

    score_label = STAT_FONT.render(
        "Score: " + str(score), True, (255, 255, 255)
    )
    win.blit(
        score_label,
        (
            WIN_WIDTH - score_label.get_width() - 15,
            10
        )
    )

    pygame.display.update()


def play_with_best_bird(config, winner_path):
    """Load winner.pkl and play with exactly one bird."""

    with open(winner_path, "rb") as file:
        winner = pickle.load(file)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    bird = Bird(230, 350)
    base = Base(FLOOR)
    pipes = [Pipe(600)]

    score = 0
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        # Select the next pipe
        pipe_ind = 0

        if (
            len(pipes) > 1
            and bird.x > pipes[0].x
            + pipes[0].PIPE_TOP.get_width()
        ):
            pipe_ind = 1

        # Use the saved winner's neural network
        output = net.activate((
            bird.y,
            abs(bird.y - pipes[pipe_ind].height),
            abs(bird.y - pipes[pipe_ind].bottom)
        ))

        if output[0] > 0.5:
            bird.jump()

        bird.move()
        base.move()

        # Move pipes and check collisions
        for pipe in pipes:
            pipe.move()

            if pipe.collide(bird, WIN):
                running = False
                break

            if (
                not pipe.passed
                and pipe.x + pipe.PIPE_TOP.get_width() < bird.x
            ):
                pipe.passed = True
                score += 1

        if not running:
            break

        # Add a new pipe when the last pipe moves far enough
        if pipes and pipes[-1].x < WIN_WIDTH - 300:
            pipes.append(Pipe(WIN_WIDTH))

        # Remove pipes that have left the screen
        pipes = [
            pipe for pipe in pipes
            if pipe.x + pipe.PIPE_TOP.get_width() > 0
        ]

        # Check floor and ceiling collisions
        if (
            bird.y + bird.img.get_height() - 10 >= FLOOR
            or bird.y < -50
        ):
            running = False

        draw_window(WIN, bird, pipes, base, score)

    pygame.quit()


def main():
    config_path = os.path.join(
        LOCAL_DIR, "config-feedforward.txt"
    )
    winner_path = os.path.join(
        LOCAL_DIR, "winner.pkl"
    )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"NEAT configuration not found: {config_path}"
        )

    if not os.path.isfile(winner_path):
        raise FileNotFoundError(
            f"Winner file not found: {winner_path}. "
            "Run your training script first to create winner.pkl."
        )

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    play_with_best_bird(config, winner_path)


if __name__ == "__main__":
    main()


