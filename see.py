import cv2
import mediapipe as mp

# تجهيز أدوات الرسم وأدوات الميديا بايب
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

# تحديد اللون الأحمر ولون الدوائر
# (B, G, R) -> (0, 0, 255) هو الأحمر الصافي
red_color = (0, 0, 255) 
drawing_spec = mp_drawing.DrawingSpec(color=red_color, thickness=1, circle_radius=1)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    # تحسين الأداء بجعل الصورة غير قابلة للكتابة مؤقتاً
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # جعل الخلفية سوداء (لتحصل على نفس تأثير الصورة التي رفعتها)
    # نقوم بمسح الصورة الأصلية وجعلها سوداء تماماً
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image[:] = (0, 0, 0) # تحويل كل البيكسلات للون الأسود

    # 1. رسم شبكة الوجه
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)

    # 2. رسم اليد اليمنى
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)

    # 3. رسم اليد اليسرى
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)

    # عرض النتيجة
    cv2.imshow('MediaPipe Holistic - Black Background', image)
    
    # الخروج عند الضغط على زر Esc
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()