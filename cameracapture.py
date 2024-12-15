import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def load_labels(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def create_class_folders(labels):
    for label in labels:
        if not os.path.exists(label):
            os.makedirs(label)

def save_image(image, folder, count):
    file_path = os.path.join(folder, f'image_{count}.jpg')
    cv2.imwrite(file_path, image)
    return file_path

class ImageCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Capture Dataset Tool")
        
        # Initialize variables
        self.labels = []
        self.selected_label = tk.StringVar()
        self.image_count = {}
        
        # Layout components
        self.label_frame = ttk.LabelFrame(root, text="Select Class")
        self.label_frame.pack(padx=10, pady=10, fill="x")
        
        self.label_dropdown = ttk.Combobox(self.label_frame, textvariable=self.selected_label, state="readonly")
        self.label_dropdown.pack(padx=5, pady=5, fill="x")
        
        self.capture_button = ttk.Button(root, text="Start Capture", command=self.start_capture)
        self.capture_button.pack(padx=10, pady=10, fill="x")
        
        self.load_labels_button = ttk.Button(root, text="Load Labels", command=self.load_labels_file)
        self.load_labels_button.pack(padx=10, pady=10, fill="x")
        
        self.camera = cv2.VideoCapture(0)

    def load_labels_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.labels = load_labels(file_path)
            create_class_folders(self.labels)
            self.label_dropdown["values"] = self.labels
            self.selected_label.set(self.labels[0])
            for label in self.labels:
                self.image_count[label] = len(os.listdir(label))
            messagebox.showinfo("Success", "Labels loaded and folders created!")

    def start_capture(self):
        if not self.labels:
            messagebox.showwarning("Warning", "Please load labels first!")
            return
        
        selected_folder = self.selected_label.get()
        if not selected_folder:
            messagebox.showwarning("Warning", "Please select a class!")
            return

        messagebox.showinfo("Info", "Press SPACE to capture images and ESC to exit.")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC to exit
                break
            elif key == 32:  # SPACE to capture
                count = self.image_count[selected_folder] + 1
                file_path = save_image(frame, selected_folder, count)
                self.image_count[selected_folder] = count
                print(f"Saved: {file_path}")

        cv2.destroyAllWindows()

    def cleanup(self):
        self.camera.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

