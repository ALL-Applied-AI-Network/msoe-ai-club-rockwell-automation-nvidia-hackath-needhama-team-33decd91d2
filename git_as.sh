#!/usr/bin/env bash

# Check if arguments were passed directly: ./git_as.sh "Name" "email"
if [ -n "$1" ] && [ -n "$2" ]; then
    GIT_NAME="$1"
    GIT_EMAIL="$2"
else
    # Prompt interactively if run with no arguments
    read -rp "Enter your Git Name (e.g., Jane Doe): " GIT_NAME
    read -rp "Enter your Git Email (e.g., jane@example.com): " GIT_EMAIL
fi

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
    echo -e "\033[0;31mError: Name and email cannot be empty.\033[0m"
    return 1 2>/dev/null || exit 1
fi

export GIT_AUTHOR_NAME="$GIT_NAME"
export GIT_COMMITTER_NAME="$GIT_NAME"
export GIT_AUTHOR_EMAIL="$GIT_EMAIL"
export GIT_COMMITTER_EMAIL="$GIT_EMAIL"

echo -e "\033[0;32m✓ Session Git identity active:\033[0m $GIT_NAME <$GIT_EMAIL>"
