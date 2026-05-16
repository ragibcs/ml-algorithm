# ভূমিকা

এই বইতে Machine Learning-এর বিভিন্ন অ্যালগরিদম সম্পর্কে বাংলায় জানতে পারবেন।

## বিষয়সমূহ

### Supervised Learning
- [সাপোর্ট ভেক্টর মেশিন (SVM)](./svm.md)
- [Random Forest](./random-forest.md)

### Unsupervised Learning
_(শীঘ্রই আসছে)_

### Reinforcement Learning
_(শীঘ্রই আসছে)_

## নতুন অ্যালগরিদম যোগ করুন

প্রতিটি অ্যালগরিদম `src/` ফোল্ডারে একটি Markdown ফাইলে রাখা হয়। নতুন অ্যালগরিদম যোগ করতে:

1. `src/` এর ভেতরে নতুন `.md` ফাইল তৈরি করুন (যেমন: `src/linear-regression.md`)
2. আর্টিকেল লিখুন
3. `src/SUMMARY.md` তে নতুন এন্ট্রি যোগ করুন
4. Push করুন — GitHub Actions স্বয়ংক্রিয়ভাবে Deploy করবে
