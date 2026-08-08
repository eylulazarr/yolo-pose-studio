import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import APP_STYLE


def main() -> int:
    app = QApplication(sys.argv)

    app.setApplicationName("YOLO Pose Studio")
    app.setOrganizationName("YOLO Pose Studio")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())