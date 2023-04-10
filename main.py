import time
import pygame
import torch
from torchvision import transforms
from cnn_model import CNNModel
from PIL import Image, ImageOps
import json

# Load the accuracy value
accuracy_save_path = "model_accuracy.json"
with open(accuracy_save_path, 'r') as f:
    accuracy_data = json.load(f)
    accuracy = accuracy_data["accuracy"]

# Train the model and get accuracy
num_epochs = 10
model_save_path = "trained_model.pt"

model = CNNModel()
model.load_state_dict(torch.load(model_save_path))
model.eval()


def crop_image(image):
    inverted_image = ImageOps.invert(image.convert('RGB'))
    box = inverted_image.getbbox()
    cropped_image = image.crop(box)
    return cropped_image


def predict_digit(surface):
    start_time = time.time()

    surface = pygame.surfarray.array3d(
        pygame.transform.rotate(
            pygame.transform.flip(surface, True, False), 90))
    surface = surface[:, :, 0]

    transform = transforms.Compose([
        transforms.Lambda(crop_image),
        transforms.Lambda(lambda x: ImageOps.expand(x, border=50, fill=255)),
        transforms.Resize((28, 28)),
        transforms.Lambda(lambda x: ImageOps.invert(x)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_pil = Image.fromarray(surface)
    img_tensor = transform(img_pil)

    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        prediction = output.argmax(dim=1).item()

    elapsed_time = time.time() - start_time
    return prediction, elapsed_time


def draw_text(screen, text, position, size=25, color=(0, 0, 0)):
    font = pygame.font.Font(None, size)
    rendered_text = font.render(text, True, color)
    screen.blit(rendered_text, position)


def draw_button(screen, text, rect, color=(0, 0, 255), text_color=(255, 255, 255)):
    pygame.draw.rect(screen, color, rect)
    draw_text(screen, text, (rect.x + 10, rect.y + 5),
              size=25, color=text_color)


pygame.init()
screen = pygame.display.set_mode((800, 280))
pygame.display.set_caption("Handwritten Digit Recognition")

background_color = (235, 235, 235)
drawing_area_color = (255, 255, 255)
screen.fill(background_color)

drawing_area = pygame.Rect(10, 10, 280, 280)
pygame.draw.rect(screen, drawing_area_color, drawing_area)

predict_button = pygame.Rect(320, 200, 100, 40)
draw_button(screen, "Predict", predict_button)

draw_text(screen, f"Model accuracy: {accuracy:.2f}%", (320, 20), size=25)

pygame.display.flip()

running = True
drawing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if drawing_area.collidepoint(event.pos):
                drawing = True
            elif predict_button.collidepoint(event.pos):
                clipped_drawing_area = drawing_area.clip(screen.get_rect())
                subsurface = screen.subsurface(clipped_drawing_area)
                digit_prediction, processing_time = predict_digit(subsurface)
                print("Predicted digit:", digit_prediction)
                screen.fill(background_color)
                pygame.draw.rect(screen, drawing_area_color, drawing_area)
                draw_button(screen, "Predict", predict_button)
                draw_text(
                    screen, f"Predicted digit: {digit_prediction}", (320, 100), size=40)
                draw_text(
                    screen, f"Processing time: {processing_time:.2f} seconds", (320, 140), size=25)
                draw_text(
                    screen, f"Model accuracy: {accuracy:.2f}%", (320, 20), size=25)
            pygame.display.flip()
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

    # Add the following lines for drawing functionality
    if drawing:
        mouse_position = pygame.mouse.get_pos()
        if drawing_area.collidepoint(mouse_position):
            pygame.draw.circle(screen, (0, 0, 0), mouse_position, 7)
            pygame.display.update(drawing_area)
            
pygame.quit()
