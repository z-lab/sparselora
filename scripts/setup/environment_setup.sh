
CONFIG=${1:-"default"}

if [ "$CONFIG" == "unsloth" ]; then
    echo "Installing spft [unsloth]..."
    pip install -e .[unsloth]
    
    pip uninstall unsloth unsloth_zoo -y && pip install --no-deps git+https://github.com/unslothai/unsloth_zoo.git && pip install --no-deps git+https://github.com/unslothai/unsloth.git

    echo "Installing flash-attn..."
    pip install flash_attn==2.7.0.post2 --no-build-isolation

else
    echo "Installing spft..."
    pip install -e .[base]


    echo "Installing flash-attn..."
    pip install flash_attn==2.7.0.post2 --no-build-isolation
fi


