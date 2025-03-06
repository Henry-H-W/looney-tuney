import pygame

# Initialize Pygame
pygame.init()

# Set screen dimensions
WIDTH, HEIGHT = 400, 150
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Button Fade-in Example")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define font
font = pygame.font.Font(None, 36)

# Define button properties
button_width, button_height = 140, 50
button1_rect = pygame.Rect(50, 50, button_width, button_height)
button2_rect = pygame.Rect(200, 50, button_width, button_height)

# Button states
button2_alpha = 0  # Fully transparent at start
button2_visible = False  # Not clickable initially
fade_in_speed = 5  # Speed of fade-in effect

# Create a transparent surface for button 2
button2_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)

# Function to render text with transparency
def render_fading_text(text, font, color, alpha):
    text_surface = font.render(text, True, color)
    text_surface.set_alpha(alpha)  # Apply transparency
    return text_surface

# Main loop
running = True
while running:
    screen.fill(BLACK)  # Set background color
    mouse_pos = pygame.mouse.get_pos()

    # Draw button 1 (Always visible and clickable)
    pygame.draw.rect(screen, BLACK, button1_rect)
    pygame.draw.rect(screen, WHITE, button1_rect, 2)
    text1 = font.render("Generation", True, WHITE)
    screen.blit(text1, (button1_rect.x + (button_width - text1.get_width()) // 2, 
                        button1_rect.y + (button_height - text1.get_height()) // 2))

    # Handle button 2 (Fade-in effect for both rectangle and text)
    if button2_visible:
        # Clear the surface and apply alpha
        button2_surface.fill((0, 0, 0, 0))  # Fully transparent background
        pygame.draw.rect(button2_surface, (255, 255, 255, button2_alpha), (0, 0, button_width, button_height), 2)

        # Render text with fading effect
        text2_surface = render_fading_text("Collaboration", font, WHITE, button2_alpha)

        # Center text inside the button surface
        button2_surface.blit(text2_surface, ((button_width - text2_surface.get_width()) // 2, 
                                             (button_height - text2_surface.get_height()) // 2))

        # Blit the transparent button onto the main screen
        screen.blit(button2_surface, (button2_rect.x, button2_rect.y))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button1_rect.collidepoint(event.pos):
                print("Generation button clicked!")
                button2_visible = True  # Start fade-in

            if button2_visible and button2_alpha >= 255 and button2_rect.collidepoint(event.pos):
                print("Collaboration button clicked!")

    # Smoothly fade in button 2 (both rectangle and text)
    if button2_visible and button2_alpha < 255:
        button2_alpha += fade_in_speed  # Increase transparency
        if button2_alpha > 255:
            button2_alpha = 255  # Cap at fully visible
        pygame.time.delay(30)  # Controls fade speed

    pygame.display.flip()  # Update display

pygame.quit()
