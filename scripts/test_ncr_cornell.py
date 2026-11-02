from falcor import *

def render_graph_NeuralRadiosity():
    g = RenderGraph("NeuralRadiosity")
    NeuralRadiosity = createPass("NeuralRadiosity", {'loadModelPath': 'media/ckpt/cornell-box/24+32_64-4.ckpt'})
    g.addPass(NeuralRadiosity, "NeuralRadiosity")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("NeuralRadiosity.color", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

NeuralRadiosity = render_graph_NeuralRadiosity()
try:
    m.addGraph(NeuralRadiosity)
    m.clock.exitFrame = 100
except NameError: None
