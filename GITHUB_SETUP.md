# How to Push This Project to GitHub

Follow these steps to get the project live on your GitHub profile.

---

## Step 1: Create a new repository on GitHub

1. Go to https://github.com/new
2. Repository name: `emotion-music-recommender`
3. Description: `AI/ML system that detects emotions from text and recommends matching music`
4. Set to **Public**
5. Do NOT initialise with README (we already have one)
6. Click **Create repository**

---

## Step 2: Initialise Git in this project folder

Open a terminal and navigate to the project folder, then run:

```bash
cd emotion-music-recommender

git init
git add .
git commit -m "Initial commit: Emotion-Based Music Recommender"
```

---

## Step 3: Link to GitHub and push

Replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/emotion-music-recommender.git
git branch -M main
git push -u origin main
```

---

## Step 4: Add a description and topics on GitHub

On your repository page:
1. Click the ⚙️ gear icon next to "About"
2. Add description: *AI that detects your emotion from text and recommends music*
3. Add topics: `machine-learning`, `nlp`, `python`, `scikit-learn`, `music`, `emotion-detection`

---

## Step 5: Verify CI is working

After pushing, click the **Actions** tab on GitHub. You should see the CI workflow running automatically. It tests across Python 3.9, 3.10, and 3.11.

---

That's it! Your project is now live. Share the link on LinkedIn or your resume. 🚀
