**פרויקט בקורס:**
**מערכות הדמיה ברפואה** – תואר שני

**מגישים:**
אליהו רוזנפלד
גיא יצחקי
---

**מאגר הקבצים – GitHub:**
[https://github.com/guyitz/xray\_streamlit\_app](https://github.com/guyitz/xray_streamlit_app)

**קבצים רלוונטיים:**

* **דוח פרויקט:** `project_report.md`
* **מחברת קוד (Jupyter Notebook):** `Xray Classification Course Project 2025.ipynb`
* **מצגת הפרויקט:** `X-Ray Image Classification.pptx`

**אתר אינטרנט להרצת המודלים (סיווג תמונות חדשות):**
[https://xray-chest-classification.streamlit.app/](https://xray-chest-classification.streamlit.app/)
*(שימו לב: אם האתר לא היה פעיל לאחרונה, ייתכן שיידרשו 1–2 דקות לטעינת המודלים. מודלי VGG16 גדולים מאוד ומשקלם כחצי ג׳יגה כל אחד).*

---

**הערה חשובה:**
כפי שצוין במהלך הצגת הפרויקט, ביצענו ניחוש מושכל לזיהוי הקטגוריות של התמונות הראשוניות. המיפוי שבחרנו הוא כדלקמן:

* `01` – **Pneumonia**
* `02` – **Normal**
* `03` – **COVID-19**

לפיכך, כאשר תוצאת המודל היא *Pneumonia*, הכוונה למחלקה `01`, וכן הלאה.

**מיקומי המודלים:**
כל המודלים נמצאים בספריית `models` במאגר ה-GitHub. המודלים הכבדים של VGG16, שגודלם רב, נמצאים בגרסאות (releases) של הפרויקט כדי להקל על הגישה וההורדה:
[https://github.com/guyitz/xray_streamlit_app/releases](https://github.com/guyitz/xray_streamlit_app/releases)