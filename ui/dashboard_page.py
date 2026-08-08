from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import (
    QEasingCurve,
    Property,
    QPropertyAnimation,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class DashboardCardData:
    route: str
    number: str
    icon: str
    title: str
    description: str
    accent_color: str
    badge: str


class NavigationCard(QFrame):
    """
    Ana sayfadaki tıklanabilir navigasyon kartı.

    Kart tıklandığında route adı ile clicked sinyali gönderilir.
    """

    clicked = Signal(str)

    def __init__(
        self,
        data: DashboardCardData,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.data = data
        self._hover_progress = 0.0

        self.setObjectName("navigationCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(205)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.setProperty("accentColor", data.accent_color)
        self.setProperty("hovered", False)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(22)
        self.shadow.setColor(QColor(0, 0, 0, 120))
        self.shadow.setOffset(0, 8)
        self.setGraphicsEffect(self.shadow)

        self.hover_animation = QPropertyAnimation(
            self,
            b"hoverProgress",
            self,
        )
        self.hover_animation.setDuration(180)
        self.hover_animation.setEasingCurve(
            QEasingCurve.Type.OutCubic
        )

        self.setup_ui()

    def setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(22, 20, 22, 20)
        root_layout.setSpacing(12)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)

        icon_container = QFrame()
        icon_container.setObjectName("cardIconContainer")
        icon_container.setFixedSize(52, 52)

        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)

        icon_label = QLabel(self.data.icon)
        icon_label.setObjectName("cardIcon")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_layout.addWidget(icon_label)

        top_text_layout = QVBoxLayout()
        top_text_layout.setContentsMargins(0, 1, 0, 0)
        top_text_layout.setSpacing(3)

        step_label = QLabel(self.data.number)
        step_label.setObjectName("cardStep")

        badge_label = QLabel(self.data.badge)
        badge_label.setObjectName("cardBadge")
        badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top_text_layout.addWidget(step_label)
        top_text_layout.addWidget(
            badge_label,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        arrow_label = QLabel("↗")
        arrow_label.setObjectName("cardArrow")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow_label.setFixedSize(38, 38)

        top_layout.addWidget(icon_container)
        top_layout.addLayout(top_text_layout)
        top_layout.addStretch()
        top_layout.addWidget(arrow_label)

        title_label = QLabel(self.data.title)
        title_label.setObjectName("dashboardCardTitle")
        title_label.setWordWrap(True)

        description_label = QLabel(self.data.description)
        description_label.setObjectName("dashboardCardDescription")
        description_label.setWordWrap(True)
        description_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )

        accent_line = QFrame()
        accent_line.setObjectName("cardAccentLine")
        accent_line.setFixedHeight(3)

        root_layout.addLayout(top_layout)
        root_layout.addSpacing(4)
        root_layout.addWidget(title_label)
        root_layout.addWidget(description_label)
        root_layout.addStretch()
        root_layout.addWidget(accent_line)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.data.route)

        super().mousePressEvent(event)

    def enterEvent(self, event) -> None:
        self.setProperty("hovered", True)
        self.style().unpolish(self)
        self.style().polish(self)

        self.hover_animation.stop()
        self.hover_animation.setStartValue(self._hover_progress)
        self.hover_animation.setEndValue(1.0)
        self.hover_animation.start()

        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.setProperty("hovered", False)
        self.style().unpolish(self)
        self.style().polish(self)

        self.hover_animation.stop()
        self.hover_animation.setStartValue(self._hover_progress)
        self.hover_animation.setEndValue(0.0)
        self.hover_animation.start()

        super().leaveEvent(event)

    def get_hover_progress(self) -> float:
        return self._hover_progress

    def set_hover_progress(self, value: float) -> None:
        self._hover_progress = value

        blur_radius = 22 + (18 * value)
        vertical_offset = 8 + (4 * value)

        self.shadow.setBlurRadius(blur_radius)
        self.shadow.setOffset(0, vertical_offset)

    hoverProgress = Property(
        float,
        get_hover_progress,
        set_hover_progress,
    )


class MetricCard(QFrame):
    def __init__(
        self,
        value: str,
        title: str,
        description: str,
        color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("metricCard")
        self.setProperty("metricColor", color)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(4)

        value_label = QLabel(value)
        value_label.setObjectName("metricValue")

        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")

        description_label = QLabel(description)
        description_label.setObjectName("metricDescription")
        description_label.setWordWrap(True)

        layout.addWidget(value_label)
        layout.addWidget(title_label)
        layout.addWidget(description_label)


class DashboardPage(QWidget):
    """
    YOLO Pose Studio modern ana sayfası.

    navigate_requested sinyali:
        annotation
        split
        augmentation
        training
        testing
    """

    navigate_requested = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("dashboardPage")
        self.setup_ui()

    def setup_ui(self) -> None:
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("dashboardScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content = QWidget()
        content.setObjectName("dashboardContent")

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(46, 34, 46, 48)
        content_layout.setSpacing(24)

        content_layout.addWidget(self.create_top_bar())
        content_layout.addWidget(self.create_hero_section())
        content_layout.addWidget(self.create_workflow_bar())

        section_header = QHBoxLayout()

        section_title_container = QVBoxLayout()
        section_title_container.setSpacing(4)

        section_label = QLabel("ÇALIŞMA ALANI")
        section_label.setObjectName("sectionEyebrow")

        section_title = QLabel("Bugün ne yapmak istiyorsun?")
        section_title.setObjectName("sectionTitle")

        section_description = QLabel(
            "Bir modül seçerek doğrudan çalışmaya başlayabilirsin."
        )
        section_description.setObjectName("sectionDescription")

        section_title_container.addWidget(section_label)
        section_title_container.addWidget(section_title)
        section_title_container.addWidget(section_description)

        section_header.addLayout(section_title_container)
        section_header.addStretch()

        content_layout.addLayout(section_header)
        content_layout.addLayout(self.create_cards_grid())
        content_layout.addWidget(self.create_footer())

        scroll_area.setWidget(content)
        page_layout.addWidget(scroll_area)

    def create_top_bar(self) -> QFrame:
        top_bar = QFrame()
        top_bar.setObjectName("dashboardTopBar")

        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(0, 0, 0, 0)

        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(12)

        logo = QLabel("YP")
        logo.setObjectName("dashboardLogo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(44, 44)

        brand_text_layout = QVBoxLayout()
        brand_text_layout.setSpacing(0)

        brand_title = QLabel("YOLO Pose Studio")
        brand_title.setObjectName("dashboardBrandTitle")

        brand_subtitle = QLabel("Dataset & Model Workspace")
        brand_subtitle.setObjectName("dashboardBrandSubtitle")

        brand_text_layout.addWidget(brand_title)
        brand_text_layout.addWidget(brand_subtitle)

        brand_layout.addWidget(logo)
        brand_layout.addLayout(brand_text_layout)

        status_container = QFrame()
        status_container.setObjectName("systemStatus")

        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(13, 7, 13, 7)
        status_layout.setSpacing(8)

        status_dot = QLabel("●")
        status_dot.setObjectName("statusDot")

        status_text = QLabel("Sistem hazır")
        status_text.setObjectName("statusText")

        status_layout.addWidget(status_dot)
        status_layout.addWidget(status_text)

        layout.addLayout(brand_layout)
        layout.addStretch()
        layout.addWidget(status_container)

        return top_bar

    def create_hero_section(self) -> QFrame:
        hero = QFrame()
        hero.setObjectName("heroSection")
        hero.setMinimumHeight(290)

        layout = QHBoxLayout(hero)
        layout.setContentsMargins(34, 32, 34, 32)
        layout.setSpacing(32)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(13)

        badge = QLabel("✦ YOLO POSE WORKSPACE")
        badge.setObjectName("heroBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedWidth(210)

        title = QLabel(
            "Pose datasetlerini\n"
            "tek merkezden yönet."
        )
        title.setObjectName("heroTitle")

        description = QLabel(
            "Etiketlemeden model testine kadar bütün YOLO Pose "
            "sürecini modern, hızlı ve kontrollü bir çalışma "
            "ortamında gerçekleştir."
        )
        description.setObjectName("heroDescription")
        description.setWordWrap(True)
        description.setMaximumWidth(610)

        supported = QLabel(
            "YOLO11 Pose  •  17 Keypoint  •  Fotoğraf  •  Video  •  Webcam"
        )
        supported.setObjectName("heroSupported")
        supported.setWordWrap(True)

        left_layout.addWidget(badge)
        left_layout.addWidget(title)
        left_layout.addWidget(description)
        left_layout.addStretch()
        left_layout.addWidget(supported)

        metrics_layout = QGridLayout()
        metrics_layout.setHorizontalSpacing(12)
        metrics_layout.setVerticalSpacing(12)

        metrics = [
            (
                "05",
                "Aktif Modül",
                "Uçtan uca pose iş akışı",
                "#2F81F7",
            ),
            (
                "17",
                "Keypoint",
                "COCO insan iskelet yapısı",
                "#A371F7",
            ),
            (
                "03",
                "Test Modu",
                "Görsel, video ve webcam",
                "#2DD4BF",
            ),
            (
                "∞",
                "Dataset",
                "Farklı YOLO Pose yapıları",
                "#F59E0B",
            ),
        ]

        for index, metric in enumerate(metrics):
            metric_card = MetricCard(*metric)
            metrics_layout.addWidget(
                metric_card,
                index // 2,
                index % 2,
            )

        layout.addLayout(left_layout, 3)
        layout.addLayout(metrics_layout, 2)

        return hero

    def create_workflow_bar(self) -> QFrame:
        workflow = QFrame()
        workflow.setObjectName("workflowBar")

        layout = QHBoxLayout(workflow)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)

        title = QLabel("İş Akışı")
        title.setObjectName("workflowTitle")

        layout.addWidget(title)
        layout.addSpacing(10)

        steps = [
            "Etiketle",
            "Böl",
            "Çoğalt",
            "Eğit",
            "Test Et",
        ]

        for index, step in enumerate(steps):
            step_container = QFrame()
            step_container.setObjectName("workflowStep")

            step_layout = QHBoxLayout(step_container)
            step_layout.setContentsMargins(10, 6, 10, 6)
            step_layout.setSpacing(7)

            number = QLabel(str(index + 1))
            number.setObjectName("workflowNumber")
            number.setAlignment(Qt.AlignmentFlag.AlignCenter)
            number.setFixedSize(23, 23)

            text = QLabel(step)
            text.setObjectName("workflowStepText")

            step_layout.addWidget(number)
            step_layout.addWidget(text)

            layout.addWidget(step_container)

            if index < len(steps) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("workflowArrow")
                layout.addWidget(arrow)

        layout.addStretch()

        return workflow

    def create_cards_grid(self) -> QGridLayout:
        cards_layout = QGridLayout()
        cards_layout.setHorizontalSpacing(18)
        cards_layout.setVerticalSpacing(18)

        cards = [
            DashboardCardData(
                route="annotation",
                number="01 / VERİ HAZIRLAMA",
                icon="⌖",
                title="Pose Etiketleme",
                description=(
                    "Görseller üzerinde bounding box oluştur, "
                    "keypoint noktalarını yerleştir ve YOLO Pose "
                    "etiketlerini profesyonel şekilde yönet."
                ),
                accent_color="#2F81F7",
                badge="ANNOTATION",
            ),
            DashboardCardData(
                route="split",
                number="02 / DATASET YÖNETİMİ",
                icon="◇",
                title="Dataset Bölme",
                description=(
                    "Görüntü ve label dosyalarını koruyarak datasetini "
                    "train, validation ve test klasörlerine dengeli "
                    "şekilde ayır."
                ),
                accent_color="#A371F7",
                badge="DATA SPLIT",
            ),
            DashboardCardData(
                route="augmentation",
                number="03 / VERİ ÇOĞALTMA",
                icon="✦",
                title="Data Augmentation",
                description=(
                    "Bounding box ve keypoint koordinatlarını koruyarak "
                    "flip, rotation, scale ve renk dönüşümleri uygula."
                ),
                accent_color="#2DD4BF",
                badge="AUGMENTATION",
            ),
            DashboardCardData(
                route="training",
                number="04 / MODEL GELİŞTİRME",
                icon="△",
                title="Model Eğitimi",
                description=(
                    "YOLO Pose modelini gelişmiş hiperparametreler, "
                    "canlı loglar, checkpoint sistemi ve validation "
                    "metrikleriyle eğit."
                ),
                accent_color="#F59E0B",
                badge="TRAINING",
            ),
            DashboardCardData(
                route="testing",
                number="05 / INFERENCE",
                icon="◎",
                title="Model Testi",
                description=(
                    "Eğitilmiş pose modelini fotoğraf, görsel klasörü, "
                    "video veya canlı webcam üzerinde gerçek zamanlı test et."
                ),
                accent_color="#F85149",
                badge="INFERENCE",
            ),
        ]

        for index, card_data in enumerate(cards):
            card = NavigationCard(card_data)
            card.clicked.connect(self.navigate_requested.emit)

            if index == len(cards) - 1:
                cards_layout.addWidget(card, 2, 0, 1, 2)
            else:
                row = index // 2
                column = index % 2
                cards_layout.addWidget(card, row, column)

        cards_layout.setColumnStretch(0, 1)
        cards_layout.setColumnStretch(1, 1)

        return cards_layout

    def create_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("dashboardFooter")

        layout = QHBoxLayout(footer)
        layout.setContentsMargins(18, 14, 18, 14)

        left_label = QLabel(
            "YOLO Pose Studio  •  Egzersiz Asistanı Geliştirme Ortamı"
        )
        left_label.setObjectName("footerText")

        right_label = QLabel("PySide6  •  Ultralytics  •  YOLO11")
        right_label.setObjectName("footerTech")

        layout.addWidget(left_label)
        layout.addStretch()
        layout.addWidget(right_label)

        return footer