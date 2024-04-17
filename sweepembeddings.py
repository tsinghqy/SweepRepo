from git import Repo

local_repo_path = "/path/to/local/repo"
django_repo_url = "https://github.com/django/django.git"

Repo.clone_from(django_repo_url, local_repo_path)