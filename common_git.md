## git clone
git clone https://github.com/XinshaoAmosWang/Deep-Metric-Embedding.git

## git add .
# Adds the file to your local repository and stages it for commit. To unstage a file, use 'git reset HEAD YOUR-FILE'.


## git commit -m "Add existing file"
# Commits the tracked changes and prepares them to be pushed to a remote repository. To remove this commit and modify the file, use 'git reset --soft HEAD~1' and commit and add the file again.

##git push origin your-branch
# Pushes the changes in your local repository up to the remote repository you specified as the origin

remote: Invalid username or password.
fatal: Authentication failed for 'https://github.com/XinshaoAmosWang/Deep-Metric-Embedding.git/'

https://stackoverflow.com/questions/29297154/github-invalid-username-or-password

git config user.name "XinshaoAmosWang"
git config user.email "xinshaowang@gmail.com"

git clone git@github.com:XinshaoAmosWang/Ranked-List-Loss-for-Deep-Metric-Learning.git

git remote set-url origin git@github.com:XinshaoAmosWang/Ranked-List-Loss-for-Deep-Metric-Learning.git

git push origin your-branch

You can now git push as normal and the correct key will automatically be used.

https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

https://www.keybits.net/post/automatically-use-correct-ssh-key-for-remote-git-repo/


## Basic settings

git config user.name "XinshaoAmosWang"

git config user.email "xinshaowang@gmail.com"

git clone git@github.com:XinshaoAmosWang/Ranked-List-Loss-for-Deep-Metric-Learning.git

git remote set-url origin git@github.com:XinshaoAmosWang/Ranked-List-Loss-for-Deep-Metric-Learning.git


## Basic commands

git add .

git commit -m "update md file"

git push origin master


## Git submodule
git submodule add git@github.com:sciencefans/CaffeMex_v2.git

git submodule init

git submodule update

git submodule update --remote --merge