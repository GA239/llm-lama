git checkout main
git fetch --all --tags --prune --prune-tags --progress -f
git pull origin
s=$(git describe --contains --all HEAD)
for a in $(git branch | tr -d " *"); do
  if [[ $s != $a ]]
  then
    git branch -D $a
  fi
done
git fetch --all --tags --prune --prune-tags --progress -f
git pull origin
git status
