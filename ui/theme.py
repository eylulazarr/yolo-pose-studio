APP_STYLE = """
/* =========================================================
   GLOBAL
========================================================= */

QWidget {
    color: #E8EDF5;
    font-family: "SF Pro Display", "Inter", "Segoe UI", Arial, sans-serif;
    font-size: 14px;
}

QMainWindow#mainWindow,
QWidget#mainWindowRoot,
QStackedWidget#mainPageStack {
    background-color: #070A10;
}

QToolTip {
    color: #F8FAFC;
    background-color: #161C27;
    border: 1px solid #344055;
    border-radius: 7px;
    padding: 7px 10px;
}

/* =========================================================
   SCROLLBAR
========================================================= */

QScrollBar:vertical {
    background: #080B12;
    width: 11px;
    margin: 4px 2px;
    border: none;
}

QScrollBar::handle:vertical {
    background: #313B4D;
    min-height: 38px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #4B5970;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
    background: transparent;
}

QScrollBar:horizontal {
    height: 0;
}

/* =========================================================
   DASHBOARD BACKGROUND
========================================================= */

QWidget#dashboardPage,
QWidget#dashboardContent {
    background-color: #070A10;
}

QScrollArea#dashboardScrollArea {
    background-color: #070A10;
    border: none;
}

/* =========================================================
   DASHBOARD TOP BAR
========================================================= */

QFrame#dashboardTopBar {
    background: transparent;
    border: none;
}

QLabel#dashboardLogo {
    color: #FFFFFF;
    background-color: #2563EB;
    border: 1px solid #60A5FA;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 800;
}

QLabel#dashboardBrandTitle {
    color: #F8FAFC;
    font-size: 19px;
    font-weight: 800;
}

QLabel#dashboardBrandSubtitle {
    color: #8492A6;
    font-size: 12px;
}

QFrame#systemStatus {
    background-color: #10251B;
    border: 1px solid #1F6F46;
    border-radius: 17px;
}

QLabel#statusDot {
    color: #3DDC84;
    font-size: 12px;
}

QLabel#statusText {
    color: #82F3B3;
    font-size: 12px;
    font-weight: 700;
}

/* =========================================================
   HERO
========================================================= */

QFrame#heroSection {
    background-color: #101623;
    border: 1px solid #263149;
    border-radius: 24px;
}

QLabel#heroBadge {
    color: #93C5FD;
    background-color: #142A4D;
    border: 1px solid #285A9F;
    border-radius: 13px;
    padding: 7px 13px;
    font-size: 11px;
    font-weight: 800;
}

QLabel#heroTitle {
    color: #FFFFFF;
    font-size: 40px;
    font-weight: 900;
}

QLabel#heroDescription {
    color: #AEB9C9;
    font-size: 15px;
}

QLabel#heroSupported {
    color: #8390A4;
    font-size: 12px;
    font-weight: 600;
}

/* =========================================================
   HERO METRICS
========================================================= */

QFrame#metricCard {
    background-color: #151C2B;
    border: 1px solid #29354C;
    border-radius: 16px;
}

QFrame#metricCard:hover {
    background-color: #192335;
    border: 1px solid #3B82F6;
}

QLabel#metricValue {
    color: #60A5FA;
    font-size: 26px;
    font-weight: 900;
}

QLabel#metricTitle {
    color: #F1F5F9;
    font-size: 13px;
    font-weight: 750;
}

QLabel#metricDescription {
    color: #8592A5;
    font-size: 11px;
}

/* =========================================================
   WORKFLOW
========================================================= */

QFrame#workflowBar {
    background-color: #0E1420;
    border: 1px solid #263149;
    border-radius: 16px;
}

QLabel#workflowTitle {
    color: #F8FAFC;
    font-size: 13px;
    font-weight: 800;
}

QFrame#workflowStep {
    background-color: #151D2B;
    border: 1px solid #2B374D;
    border-radius: 11px;
}

QLabel#workflowNumber {
    color: #FFFFFF;
    background-color: #2563EB;
    border-radius: 11px;
    font-size: 11px;
    font-weight: 800;
}

QLabel#workflowStepText {
    color: #CAD3E0;
    font-size: 12px;
    font-weight: 650;
}

QLabel#workflowArrow {
    color: #657187;
    font-size: 16px;
}

/* =========================================================
   SECTION HEADERS
========================================================= */

QLabel#sectionEyebrow {
    color: #60A5FA;
    font-size: 11px;
    font-weight: 850;
}

QLabel#sectionTitle {
    color: #F8FAFC;
    font-size: 28px;
    font-weight: 900;
}

QLabel#sectionDescription {
    color: #8592A5;
    font-size: 13px;
}

/* =========================================================
   NAVIGATION CARDS
========================================================= */

QFrame#navigationCard {
    background-color: #101722;
    border: 1px solid #283449;
    border-radius: 20px;
}

QFrame#navigationCard:hover,
QFrame#navigationCard[hovered="true"] {
    background-color: #162033;
    border: 1px solid #4C8DFF;
}

QFrame#cardIconContainer {
    background-color: #13294D;
    border: 1px solid #285A9F;
    border-radius: 14px;
}

QLabel#cardIcon {
    color: #70B5FF;
    font-size: 25px;
    font-weight: 900;
}

QLabel#cardStep {
    color: #8492A6;
    font-size: 10px;
    font-weight: 800;
}

QLabel#cardBadge {
    color: #C5D0DF;
    background-color: #1B2433;
    border: 1px solid #354158;
    border-radius: 8px;
    padding: 3px 8px;
    font-size: 9px;
    font-weight: 800;
}

QLabel#cardArrow {
    color: #92A1B5;
    background-color: #192230;
    border: 1px solid #354158;
    border-radius: 11px;
    font-size: 18px;
    font-weight: 800;
}

QFrame#navigationCard:hover QLabel#cardArrow,
QFrame#navigationCard[hovered="true"] QLabel#cardArrow {
    color: #FFFFFF;
    background-color: #2563EB;
    border: 1px solid #60A5FA;
}

QLabel#dashboardCardTitle {
    color: #F8FAFC;
    font-size: 21px;
    font-weight: 850;
}

QLabel#dashboardCardDescription {
    color: #9DAABC;
    font-size: 13px;
}

QFrame#cardAccentLine {
    background-color: #3B82F6;
    border-radius: 2px;
}

/* =========================================================
   FOOTER
========================================================= */

QFrame#dashboardFooter {
    background-color: #0E1420;
    border: 1px solid #263149;
    border-radius: 14px;
}

QLabel#footerText,
QLabel#footerTech {
    color: #718096;
    font-size: 11px;
}

/* =========================================================
   TOP NAVIGATION
========================================================= */

QFrame#topNavigationBar {
    background-color: #0C111B;
    border: none;
    border-bottom: 1px solid #263149;
}

QPushButton#backButton {
    color: #E5EDF8;
    background-color: #172131;
    border: 1px solid #344158;
    border-radius: 11px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 700;
}

QPushButton#backButton:hover {
    color: #FFFFFF;
    background-color: #25334A;
    border-color: #4C8DFF;
}

QPushButton#backButton:pressed {
    background-color: #101722;
}

QLabel#topNavigationTitle {
    color: #F8FAFC;
    font-size: 18px;
    font-weight: 850;
}

QLabel#topNavigationSubtitle {
    color: #8390A4;
    font-size: 11px;
}

QFrame#topStatusBadge {
    background-color: #10251B;
    border: 1px solid #1F6F46;
    border-radius: 16px;
}

QLabel#topStatusDot {
    color: #3DDC84;
}

QLabel#topStatusText {
    color: #82F3B3;
    font-size: 12px;
    font-weight: 700;
}

/* =========================================================
   GENERIC CONTROLS
========================================================= */

QPushButton {
    color: #FFFFFF;
    background-color: #2563EB;
    border: 1px solid #4C8DFF;
    border-radius: 9px;
    padding: 9px 15px;
    font-weight: 700;
}

QPushButton:hover {
    background-color: #3678F4;
}

QPushButton:pressed {
    background-color: #1D4ED8;
}

QPushButton:disabled {
    color: #677386;
    background-color: #171D28;
    border-color: #2B3443;
}

QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {
    color: #E8EDF5;
    background-color: #101722;
    border: 1px solid #344158;
    border-radius: 8px;
    min-height: 37px;
    padding-left: 10px;
    padding-right: 10px;
    selection-background-color: #2563EB;
}

QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {
    border: 1px solid #4C8DFF;
}

QTextEdit,
QPlainTextEdit {
    color: #D8E0EB;
    background-color: #080C13;
    border: 1px solid #283449;
    border-radius: 10px;
    selection-background-color: #2563EB;
}

QCheckBox {
    color: #D7DFEA;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 17px;
    height: 17px;
    background-color: #101722;
    border: 1px solid #46536A;
    border-radius: 4px;
}

QCheckBox::indicator:checked {
    background-color: #2563EB;
    border-color: #60A5FA;
}

QGroupBox {
    color: #F1F5F9;
    background-color: #0E1420;
    border: 1px solid #283449;
    border-radius: 13px;
    margin-top: 14px;
    padding-top: 15px;
    font-weight: 750;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding-left: 6px;
    padding-right: 6px;
}

QTabWidget::pane {
    background-color: #0E1420;
    border: 1px solid #283449;
    border-radius: 10px;
}

QTabBar::tab {
    color: #909DB0;
    background-color: #151D2B;
    border: none;
    padding: 10px 18px;
}

QTabBar::tab:selected {
    color: #FFFFFF;
    background-color: #2563EB;
}
/* =========================================================
   DATASET SPLITTER PAGE
========================================================= */

QWidget#splitterPage,
QWidget#splitterContent,
QScrollArea#splitterScrollArea {
    background-color: #070A10;
}

QFrame#pageIntroFrame {
    background: transparent;
    border: none;
}

QLabel#pageTitle {
    color: #F8FAFC;
    font-size: 31px;
    font-weight: 900;
}

QLabel#pageDescription {
    color: #9AA8BA;
    font-size: 14px;
}

QFrame#sectionCard {
    background-color: #0E1420;
    border: 1px solid #283449;
    border-radius: 17px;
}

QFrame#sectionCard QLabel {
    background: transparent;
}

QLabel#formLabel {
    color: #D7DFEA;
    font-size: 13px;
    font-weight: 650;
}

QPushButton#pathSelectButton {
    color: #FFFFFF;
    background-color: #2563EB;
    border: 1px solid #4C8DFF;
    border-radius: 9px;
    padding: 0 12px;
    font-weight: 700;
}

QPushButton#pathSelectButton:hover {
    background-color: #3678F4;
}

QPushButton#primaryButton {
    background-color: #2563EB;
    border: 1px solid #60A5FA;
}

QPushButton#primaryButton:hover {
    background-color: #3B82F6;
}

QPushButton#secondaryButton {
    color: #DCE6F2;
    background-color: #172131;
    border: 1px solid #344158;
}

QPushButton#secondaryButton:hover {
    color: #FFFFFF;
    background-color: #233149;
    border-color: #4C8DFF;
}

QLabel#operationStatusLabel {
    color: #97A5B8;
    font-size: 13px;
}

QLabel#ratioStatusLabel {
    color: #F59E0B;
    font-size: 12px;
    font-weight: 650;
}

QLabel#ratioStatusLabel[valid="true"] {
    color: #4ADE80;
}

QLabel#ratioStatusLabel[valid="false"] {
    color: #F59E0B;
}

QProgressBar#splitProgressBar {
    color: #E8EDF5;
    background-color: #101722;
    border: 1px solid #344158;
    border-radius: 8px;
    text-align: center;
}

QProgressBar#splitProgressBar::chunk {
    background-color: #2563EB;
    border-radius: 7px;
}

QTextEdit#splitResultText {
    color: #D7E0EC;
    background-color: #080C13;
    border: 1px solid #283449;
    border-radius: 11px;
    padding: 10px;
    font-family: "SF Mono", "Menlo", monospace;
    font-size: 12px;
}

"""