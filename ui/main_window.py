from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# Paket içindeki dosyalarda relative import kullanıyoruz.
from .annotation_page import AnnotationPage
from .augmentation_page import AugmentationPage
from .dashboard_page import DashboardPage
from .splitter_page import SplitterPage
from .testing_page import TestingPage
from .training_page import TrainingPage


class TopNavigationBar(QFrame):
    """Alt sayfalarda gösterilen üst navigasyon çubuğu."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setObjectName("topNavigationBar")
        self.setFixedHeight(76)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 12, 28, 12)
        layout.setSpacing(16)

        self.back_button = QPushButton("←  Ana Sayfa")
        self.back_button.setObjectName("backButton")
        self.back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_button.setFixedHeight(42)

        title_container = QVBoxLayout()
        title_container.setContentsMargins(0, 0, 0, 0)
        title_container.setSpacing(2)

        self.title_label = QLabel("YOLO Pose Studio")
        self.title_label.setObjectName("topNavigationTitle")

        self.subtitle_label = QLabel("Çalışma alanı")
        self.subtitle_label.setObjectName("topNavigationSubtitle")

        title_container.addWidget(self.title_label)
        title_container.addWidget(self.subtitle_label)

        status_badge = QFrame()
        status_badge.setObjectName("topStatusBadge")

        status_layout = QHBoxLayout(status_badge)
        status_layout.setContentsMargins(12, 7, 12, 7)
        status_layout.setSpacing(7)

        status_dot = QLabel("●")
        status_dot.setObjectName("topStatusDot")

        status_text = QLabel("YOLO Pose hazır")
        status_text.setObjectName("topStatusText")

        status_layout.addWidget(status_dot)
        status_layout.addWidget(status_text)

        layout.addWidget(self.back_button)
        layout.addLayout(title_container)
        layout.addStretch()
        layout.addWidget(status_badge)

    def set_page_information(self, title: str, subtitle: str) -> None:
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)


class MainWindow(QMainWindow):
    """Sidebar olmadan sayfa geçişlerini yöneten ana pencere."""

    PAGE_INFORMATION = {
        "annotation": (
            "Pose Etiketleme",
            "Bounding box ve keypoint etiketlerini oluştur.",
        ),
        "split": (
            "Dataset Bölme",
            "Datasetini train, validation ve test olarak ayır.",
        ),
        "augmentation": (
            "Data Augmentation",
            "Görselleri ve pose koordinatlarını birlikte dönüştür.",
        ),
        "training": (
            "Model Eğitimi",
            "YOLO Pose modelini yapılandır ve eğit.",
        ),
        "testing": (
            "Model Testi",
            "Modelini fotoğraf, video ve webcam üzerinde test et.",
        ),
    }

    def __init__(self) -> None:
        super().__init__()

        self.setObjectName("mainWindow")
        self.setWindowTitle("YOLO Pose Studio")

        self.setMinimumSize(1120, 720)
        self.resize(1440, 900)

        self.setup_ui()
        self.connect_signals()
        self.show_dashboard()

    def setup_ui(self) -> None:
        root = QWidget()
        root.setObjectName("mainWindowRoot")

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.top_navigation = TopNavigationBar()
        self.top_navigation.hide()

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("mainPageStack")
        self.page_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.dashboard_page = DashboardPage()
        self.annotation_page = AnnotationPage()
        self.split_page = SplitterPage()
        self.augmentation_page = AugmentationPage()
        self.training_page = TrainingPage()
        self.testing_page = TestingPage()

        self.pages: dict[str, QWidget] = {
            "dashboard": self.dashboard_page,
            "annotation": self.annotation_page,
            "split": self.split_page,
            "augmentation": self.augmentation_page,
            "training": self.training_page,
            "testing": self.testing_page,
        }

        for page in self.pages.values():
            self.page_stack.addWidget(page)

        root_layout.addWidget(self.top_navigation)
        root_layout.addWidget(self.page_stack, 1)

        self.setCentralWidget(root)

    def connect_signals(self) -> None:
        self.dashboard_page.navigate_requested.connect(self.show_page)
        self.top_navigation.back_button.clicked.connect(self.show_dashboard)

    def show_dashboard(self) -> None:
        self.top_navigation.hide()
        self.page_stack.setCurrentWidget(self.dashboard_page)

    def show_page(self, route: str) -> None:
        page = self.pages.get(route)

        if page is None:
            print(f"Bilinmeyen sayfa route'u: {route}")
            return

        title, subtitle = self.PAGE_INFORMATION.get(
            route,
            ("YOLO Pose Studio", "Çalışma alanı"),
        )

        self.top_navigation.set_page_information(title, subtitle)
        self.top_navigation.show()
        self.page_stack.setCurrentWidget(page)