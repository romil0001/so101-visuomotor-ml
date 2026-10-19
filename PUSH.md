# Publish to GitHub

This folder is ready. System `git` is blocked until Xcode license is accepted
(`sudo xcodebuild -license`). Then:

```bash
cd ~/Developer/so101-visuomotor-ml
gh auth refresh -h github.com   # if needed
git init -b main
git add -A
git commit -m "Initial SO-101 visuomotor ML showcase"
gh repo create so101-visuomotor-ml --public --source=. --remote=origin --push
```

Or create an empty repo on github.com named `so101-visuomotor-ml`, then:

```bash
git remote add origin git@github.com:romil0001/so101-visuomotor-ml.git
git push -u origin main
```
