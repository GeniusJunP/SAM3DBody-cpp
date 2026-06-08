import time
import coremltools as ct
import numpy as np

mlmodel = ct.models.MLModel('coreml_export/decoder_coreml.mlpackage')

features = np.random.randn(1, 1280, 32, 32).astype(np.float32)
condition_info = np.random.randn(1, 3).astype(np.float32)
ray_cond = np.random.randn(1, 2, 32, 32).astype(np.float32)

inputs = {
    'features': features,
    'condition_info': condition_info,
    'ray_cond': ray_cond
}

# Warmup
for _ in range(5):
    mlmodel.predict(inputs)

# Measure
times = []
for _ in range(50):
    t0 = time.time()
    mlmodel.predict(inputs)
    times.append(time.time() - t0)

avg_ms = np.mean(times) * 1000
fps = 1000 / avg_ms
print(f"CoreML Decoder Performance: {avg_ms:.2f} ms / inference ({fps:.2f} FPS)")
