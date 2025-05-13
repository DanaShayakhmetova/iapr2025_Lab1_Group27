import cv2
import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    @staticmethod
    def show_image(image, title="Image", size=(6,6)):
        plt.figure(figsize=size)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def show_image_with_hsv(hsv_image, title="HSV Hover", size=(10, 10)):
        """Display an HSV image and show HSV value at the mouse cursor position"""
        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(hsv_image)
        ax.set_title(title)
        ax.axis('off')

        # Text box to display HSV values
        text = ax.text(10, 10, "", color='white', fontsize=12,
                       bbox=dict(facecolor='black', alpha=0.5))

        def format_coord(x, y):
            """
            Get HSV values from pixel at (x, y).
            """
            x, y = int(x), int(y)
            if 0 <= x < hsv_image.shape[1] and 0 <= y < hsv_image.shape[0]:
                h, s, v = hsv_image[y, x]
                return f"x={x}, y={y}, HSV=({h}, {s}, {v})"
            else:
                return ""

        def on_mouse_move(event):
            """
            Update HSV text when mouse moves over the image.
            """
            if event.inaxes == ax and event.xdata and event.ydata:
                msg = format_coord(event.xdata, event.ydata)
                text.set_text(msg)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
        plt.show()

    def draw_detected_circle(image, circles):
        """Draw the circles found by find_Hough_circle function for visualization"""
        output = image.copy()

        if circles is not None:
            circles = np.int16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 8)) 
        plt.imshow(output_rgb)
        plt.axis('off')
        plt.title("Detected Circles")
        plt.show()