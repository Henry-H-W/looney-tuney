import sys
import pygame
import rtmidi
import random
import colorsys
from explosion_animation import Firework  # Make sure game.py is in the same directory or python path

def main():
    # --- 1) Initialize Pygame and the MIDI system ---
    pygame.init()
    midi_in = rtmidi.MidiIn()

    # List available MIDI input ports
    ports = midi_in.get_ports()
    if not ports:
        print("No MIDI input devices found.")
        return

    print("Available MIDI ports:")
    for i, port in enumerate(ports):
        print(f"{i}: {port}")

    # Prompt user for MIDI port selection
    port_number = int(input("Select MIDI input port number: "))
    if port_number < 0 or port_number >= len(ports):
        print("Invalid port number.")
        return

    # Open the chosen MIDI port
    midi_in.open_port(port_number)

    # --- 2) Set up the Pygame window ---
    screen_width = 640
    screen_height = 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Firework Animation via MIDI Keyboard")
    clock = pygame.time.Clock()

    # --- 3) Prepare to track fireworks ---
    fireworks = []        # holds active Firework objects
    hue_index = 0.0       # used for cycling firework colors

    # Create a small "parent" object so Firework can access window size
    class Parent:
        pass

    parent_obj = Parent()
    parent_obj.window_size = (screen_width, screen_height)

    # --- 4) Main Loop ---
    running = True
    while running:
        # A) Handle pygame events (quit, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # B) Poll MIDI input
        msg = midi_in.get_message()
        if msg:
            message, delta_time = msg
            # Typically: message = [status, note, velocity, ...]
            if len(message) >= 3:
                note = message[1]
                velocity = message[2]
                # If velocity > 0 => "note on"
                if velocity > 0:
                    # Convert MIDI note (range 36..96) to horizontal position [0..1]
                    x_pos = (note - 36) / float(96 - 36)
                    # Clamp or adjust if needed
                    if x_pos < 0: x_pos = 0
                    if x_pos > 1: x_pos = 1

                    # Convert HSV to RGB for the rainbow color
                    color = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(hue_index, 1.0, 1.0))

                    # Create a new firework at that position
                    fw = Firework(parent_obj, x_pos, color, intensity=1)
                    fireworks.append(fw)

                    # Cycle the hue for the next firework
                    hue_index += 0.08
                    if hue_index > 1.0:
                        hue_index = 0.0

        # C) Update all fireworks
        for fw in fireworks:
            fw.Update()

        # Remove any finished fireworks
        fireworks = [fw for fw in fireworks if fw.state != 'finished']

        # D) Draw everything
        screen.fill((0, 0, 0))  # black background
        for fw in fireworks:
            fw.Draw(screen)

        pygame.display.flip()
        clock.tick(30)  # ~30 FPS

    # --- 5) Cleanup ---
    midi_in.close_port()
    pygame.quit()

if __name__ == "__main__":
    main()
