import pygame

# Initialize Pygame
pygame.init()

# Set screen dimensions
WIDTH, HEIGHT = 400, 150
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Button Toggle Example")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)

# Define font
font = pygame.font.Font(None, 36)

# Define button properties
button_width, button_height = 140, 50
button1_rect = pygame.Rect(50, 50, button_width, button_height)
button2_rect = pygame.Rect(200, 50, button_width, button_height)

# Track which button is active
active_button = None  # Can be "generation" or "collaboration"

# Main loop
running = True
while running:
    screen.fill(BLACK)  # Set background color

    # Draw buttons with dynamic colors based on selection
    if active_button == "generation":
        pygame.draw.rect(screen, WHITE, button1_rect)  # White background
        pygame.draw.rect(screen, WHITE, button1_rect, 2)  # Border
        text1_color = BLACK  # Black text
    else:
        pygame.draw.rect(screen, BLACK, button1_rect)  # Black background
        pygame.draw.rect(screen, WHITE, button1_rect, 2)  # White border
        text1_color = WHITE  # White text

    if active_button == "collaboration":
        pygame.draw.rect(screen, WHITE, button2_rect)  # White background
        pygame.draw.rect(screen, WHITE, button2_rect, 2)  # Border
        text2_color = BLACK  # Black text
    else:
        pygame.draw.rect(screen, BLACK, button2_rect)  # Black background
        pygame.draw.rect(screen, WHITE, button2_rect, 2)  # White border
        text2_color = WHITE  # White text

    # Render text
    text1 = font.render("Generation", True, text1_color)
    text2 = font.render("Collaboration", True, text2_color)

    # Center text inside buttons
    screen.blit(text1, (button1_rect.x + (button_width - text1.get_width()) // 2, button1_rect.y + (button_height - text1.get_height()) // 2))
    screen.blit(text2, (button2_rect.x + (button_width - text2.get_width()) // 2, button2_rect.y + (button_height - text2.get_height()) // 2))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button1_rect.collidepoint(event.pos):
                active_button = "generation"
            elif button2_rect.collidepoint(event.pos):
                active_button = "collaboration"

    pygame.display.flip()  # Update display

pygame.quit()
