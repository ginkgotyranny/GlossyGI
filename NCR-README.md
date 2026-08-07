# Neural Radiosity RenderPass

To run this module, we need **Falcor** and **tiny-cuda-nn** properly compiled. It should be ready if you are in this git branch: `ncr`.

`NeuralRadiosity` pipeline has been tested on Ubuntu 24.04 with Vulkan backend. (We did not test on D3D12, but since there is no platform-specific code in this module, it is expected to work soundly as long as Falcor is ready.)

### Compiling Falcor

* `git submodule update --init --recursive`
* Follow Falcor's `README.md`:
* `cmake --preset linux-gcc -DCMAKE_BUILD_TYPE=Release`
* `cmake --build build/linux-gcc`

### Rendering with NeuralRadiosity

* Run `./run.sh`
* Click "NeuralRadiosity", load models from `media/ckpt`. We have `living-room` and `kitchen` trained already.
* Modify the scene directory in `run.sh` accordingly.

Online training / dynamic scene:
* Run `./animation`.
* Click "NeuralRadiosity", choose render mode: "Online train".
* Wait for the radiance distribution to converge.
* The scene in this demo will automatically animate.

### Training a model

* Run `./run.sh`.
* Click "NeuralRadiosity", choose render mode: "Train".
* The training process will start and the steps are shown.
* Upon training finished, the render mode would switch to "Idle". Checkpoints are saved in the main directory.