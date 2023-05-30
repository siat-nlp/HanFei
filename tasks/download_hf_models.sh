set -e

git lfs install

CURRENT_DIR=$(dirname $(readlink -f "$0"))
BASE_DIR=$(dirname ${CURRENT_DIR})
TARGET_DIR="${BASE_DIR}/resources/external_models"

cd ${TARGET_DIR}
git clone https://huggingface.co/facebook/bart-base