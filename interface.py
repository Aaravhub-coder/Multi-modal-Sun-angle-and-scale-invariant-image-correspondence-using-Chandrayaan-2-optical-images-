### THIS FILE CONTAINS THE UI INTERFACE FOR THE PROJECT. IT IS RESPONSIBLE FOR HANDLING USER INPUT AND DISPLAYING OUTPUT. ###
"""
Minimal interface with two buttons: "Source Image" and "Reference".
Clicking either button opens the OS file manager (native file dialog)
so the user can pick a file. The chosen paths are stored in
`source_image_path` and `reference_path` and printed to the console,
ready to be used as input elsewhere in your program.

Requires: pygame  (pip install pygame)
tkinter is used only for the native "Open File" dialog and is part of
the Python standard library on most installations.
"""

import pygame
import sys
import os
from tkinter import Tk, filedialog

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 480, 260
BG_COLOR = (30, 30, 30)
BUTTON_COLOR = (60, 60, 60)
BUTTON_HOVER_COLOR = (90, 90, 90)
TEXT_COLOR = (255, 255, 255)
PATH_TEXT_COLOR = (150, 220, 150)
FONT_NAME = None  # default pygame font

# These will hold the file paths chosen by the user.
# Use them anywhere else in your program.
source_image_path = None
reference_path = None


# ---------------------------------------------------------------------------
# Helper: open a native "choose file" dialog and return the selected path
# ---------------------------------------------------------------------------
def open_file_dialog(title="Select a file"):
    root = Tk()
    root.withdraw()          # hide the empty tkinter root window
    root.attributes("-topmost", True)  # bring dialog to front
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path if file_path else None


# ---------------------------------------------------------------------------
# Simple Button class
# ---------------------------------------------------------------------------
class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.hovered = False

    def draw(self, surface, font):
        color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, (120, 120, 120), self.rect, width=2, border_radius=8)

        text_surf = font.render(self.label, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos, mouse_click):
        return self.rect.collidepoint(mouse_pos) and mouse_click


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global source_image_path, reference_path

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Select Inputs")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(FONT_NAME, 24)
    small_font = pygame.font.SysFont(FONT_NAME, 18)

    button_width, button_height = 180, 60
    gap = 30
    total_width = button_width * 2 + gap
    start_x = (WIDTH - total_width) // 2
    y = 60

    source_button = Button((start_x, y, button_width, button_height), "Source Image")
    reference_button = Button((start_x + button_width + gap, y, button_width, button_height), "Reference")

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicked = True

        source_button.update_hover(mouse_pos)
        reference_button.update_hover(mouse_pos)

        if source_button.is_clicked(mouse_pos, mouse_clicked):
            path = open_file_dialog(title="Select Source Image")
            if path:
                source_image_path = path
                print(f"[Source Image] selected: {source_image_path}")

        if reference_button.is_clicked(mouse_pos, mouse_clicked):
            path = open_file_dialog(title="Select Reference File")
            if path:
                reference_path = path
                print(f"[Reference] selected: {reference_path}")

        # -------------------- Draw --------------------
        screen.fill(BG_COLOR)

        source_button.draw(screen, font)
        reference_button.draw(screen, font)

        # Show selected file names (truncated) below the buttons
        src_label = os.path.basename(source_image_path) if source_image_path else "No file selected"
        ref_label = os.path.basename(reference_path) if reference_path else "No file selected"

        src_surf = small_font.render(f"Source: {src_label}", True, PATH_TEXT_COLOR)
        ref_surf = small_font.render(f"Reference: {ref_label}", True, PATH_TEXT_COLOR)

        screen.blit(src_surf, (20, 160))
        screen.blit(ref_surf, (20, 190))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

    # -------------------- Use the paths here --------------------
    print("\nFinal selections:")
    print("Source image path:", source_image_path)
    print("Reference path:", reference_path)

    # e.g. pass them on to the rest of your program:
    # run_my_program(source_image_path, reference_path)


if __name__ == "__main__":
    main()