"""Update docker-compose bind mounts based on user-supplied host directories."""

import json  # Added to parse the staged directory list from Flask.
import os  # Added to remove temporary files once they are consumed.
import subprocess  # Added to restart Docker so new mounts apply immediately.
import sys  # Added to check CLI arguments and exit with guidance.
from pathlib import Path  # Added to normalise host directory paths consistently.

import yaml  # Added dependency to rewrite docker-compose.yml safely.

APP_SERVICE = 'app'  # Added constant to avoid repeating the service name string.
TARGET_TEMPLATE = '/mnt/data{index}'  # Added template so mount targets remain predictable.


def load_directories(config_path):
    """Return a list of host directories from the temporary JSON file."""

    with open(config_path, 'r', encoding='utf-8') as handle:
        # Added JSON loading to retrieve the directories staged by the Flask endpoint.
        directories = json.load(handle)
    try:
        os.unlink(config_path)
    except OSError:
        # Added best-effort cleanup so leftover temporary files do not accumulate.
        pass
    return [str(Path(item).expanduser()) for item in directories]


def load_compose(compose_path):
    """Load docker-compose.yml into a Python dictionary."""

    with open(compose_path, 'r', encoding='utf-8') as handle:
        # Added YAML parsing to work with the compose definition programmatically.
        return yaml.safe_load(handle)


def dump_compose(compose_path, payload):
    """Write the modified compose data back to disk."""

    with open(compose_path, 'w', encoding='utf-8') as handle:
        # Added YAML dump to persist the updated volume definitions with stable ordering.
        yaml.safe_dump(payload, handle, sort_keys=False)


def filter_static_volumes(volumes):
    """Remove any previously generated /mnt/data* mounts from the volume list."""

    static_volumes = []
    for entry in volumes or []:
        target = None
        if isinstance(entry, dict):
            target = entry.get('target')
        elif isinstance(entry, str) and ':' in entry:
            # Added support for string-based volume syntax to avoid dropping manual mounts.
            _, _, target = entry.partition(':')
        if target and target.strip().startswith('/mnt/data'):
            # Added skip so dynamic mounts are replaced cleanly on each update.
            continue
        static_volumes.append(entry)
    return static_volumes


def build_mounts(directories):
    """Create volume entries for each host directory."""

    new_volumes = []
    for index, directory in enumerate(directories, start=1):
        if not directory:
            # Added guard to ignore empty strings that might slip through validation.
            continue
        new_volumes.append({
            'type': 'bind',
            'source': directory,
            'target': TARGET_TEMPLATE.format(index=index),
        })
    return new_volumes


def update_volumes(compose_data, directories):
    """Update the app service volumes with static and dynamic mounts."""

    services = compose_data.setdefault('services', {})
    if APP_SERVICE not in services:
        # Added explicit error so the script fails loudly if the expected service is missing.
        raise KeyError('app service not found in docker-compose.yml')

    app_service = services[APP_SERVICE]
    static_volumes = filter_static_volumes(app_service.get('volumes', []))
    dynamic_volumes = build_mounts(directories)
    # Added concatenation so required static mounts remain alongside the new data mounts.
    app_service['volumes'] = static_volumes + dynamic_volumes


def rebuild_docker():
    """Restart the Docker stack so new mounts take effect."""

    # Added compose down to release any bind mounts that might conflict with the new set.
    subprocess.run(['docker', 'compose', 'down'], check=False)
    # Added compose up --build to rebuild the app container with the refreshed mounts.
    subprocess.run(['docker', 'compose', 'up', '-d', '--build'], check=True)


def update_mounts(config_path, compose_path='docker-compose.yml'):
    """Main entry point for refreshing docker-compose mounts."""

    directories = load_directories(config_path)
    compose_data = load_compose(compose_path)
    update_volumes(compose_data, directories)
    dump_compose(compose_path, compose_data)
    rebuild_docker()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Added usage hint so manual invocations receive guidance.
        print('Usage: python update_mounts.py <config_path>')
        sys.exit(1)

    update_mounts(sys.argv[1])
