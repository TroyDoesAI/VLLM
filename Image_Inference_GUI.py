def start_gui():
    import sys
    try:
        from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QTextEdit
        from PyQt5.QtGui import QPixmap
        from VLLM import VLLM  # Import the VLLM class
    except ImportError as e:
        print("Failed to import PyQt5 or VLLM module:", e)
        return

    class ImageInferenceApp(QWidget):
        def __init__(self):
            super().__init__()

            self.setWindowTitle('Image Inference App')

            self.vllm = VLLM()

            self.image_label = QLabel(self)
            self.image_label.setFixedSize(300, 300)

            self.select_image_button = QPushButton('Select Image')
            self.select_image_button.clicked.connect(self.select_image)

            self.question_textedit = QTextEdit()
            self.question_textedit.setPlaceholderText('Enter your question')

            self.infer_button = QPushButton('Infer')
            self.infer_button.clicked.connect(self.perform_inference)

            self.output_label = QLabel()
            self.output_label.setWordWrap(True)

            layout = QVBoxLayout()
            layout.addWidget(self.image_label)
            layout.addWidget(self.select_image_button)
            layout.addWidget(self.question_textedit)
            layout.addWidget(self.infer_button)
            layout.addWidget(self.output_label)

            self.setLayout(layout)

        def select_image(self):
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            file_dialog.setViewMode(QFileDialog.Detail)
            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                if file_paths:
                    self.image_path = file_paths[0]
                    pixmap = QPixmap(self.image_path)
                    self.image_label.setPixmap(pixmap.scaled(300, 300))

        def perform_inference(self):
            if hasattr(self, 'image_path'):
                question = self.question_textedit.toPlainText()
                query = f"Provided image: {self.image_path}\n{question}"
                result = self.vllm.infer(self.image_path, query)
                self.output_label.setText(result)
            else:
                print("Please select an image first.")

    app = QApplication(sys.argv)
    window = ImageInferenceApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    start_gui()
