
CONFIG=${1:-"default"}

if [ "$CONFIG" == "unsloth" ]; then
    echo "Installing spft [unsloth]..."
    pip install -e .[unsloth]
    
    pip uninstall unsloth unsloth_zoo -y && \
    pip install --no-deps unsloth_zoo==2025.6.7 && \
    pip install --no-deps unsloth==2025.6.9

    #* Alternatively use latest version, may hit some conflicts
    # pip uninstall unsloth unsloth_zoo -y && pip install --no-deps git+https://github.com/unslothai/unsloth_zoo.git && pip install --no-deps git+https://github.com/unslothai/unsloth.git

    pip install pillow==11.2.1

    echo "Installing flash-attn..."
    pip install flash_attn==2.7.0.post2 --no-build-isolation

else
    echo "Installing spft..."
    pip install -e .[base]


    echo "Installing flash-attn..."
    pip install flash_attn==2.7.0.post2 --no-build-isolation
fi


