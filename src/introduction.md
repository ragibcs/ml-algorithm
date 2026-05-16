# ML Algorithm Book

## ভাষা নির্বাচন / Language Selection

---

### 🇧🇩 [বাংলায় পড়ুন (Read in Bangla)](./svm.md)

Machine Learning অ্যালগরিদম সম্পর্কে বাংলায় জানুন।

### 🇬🇧 [Read in English](./en/svm.md)

Learn about Machine Learning algorithms in English.

---

## বিষয়সমূহ / Topics

| Algorithm | বাংলা | English |
| --------- | ----- | ------- |
| SVM | [পড়ুন](./svm.md) | [Read](./en/svm.md) |
| Random Forest | [পড়ুন](./random-forest.md) | [Read](./en/random-forest.md) |
| XGBoost | [পড়ুন](./xgboost.md) | [Read](./en/xgboost.md) |

---

## নতুন অ্যালগরিদম যোগ করুন / Add New Algorithm

প্রতিটি অ্যালগরিদম `src/` ফোল্ডারে একটি Markdown ফাইলে রাখা হয়। / Each algorithm is a single `.md` file in the `src/` folder.

1. `src/` তে নতুন `.md` ফাইল তৈরি করুন (বাংলা) / Create a new `.md` file in `src/` (Bangla)
2. `src/en/` তে ইংরেজি সংস্করণ তৈরি করুন / Create English version in `src/en/`
3. `src/SUMMARY.md` তে এন্ট্রি যোগ করুন / Add entry in `src/SUMMARY.md`
4. Push করুন — GitHub Actions স্বয়ংক্রিয়ভাবে Deploy করবে / Push — GitHub Actions auto-deploys
