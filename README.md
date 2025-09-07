# Vapoursynth-Deinterlace-CUDA-Test
まだテスト段階。BwdifをCUDAで再現するプロジェクト。

```
clip1 = core.std.AssumeFPS(clip   , fpsnum=60000, fpsden=1001)


deint = core.cdeint.CudaDeinterlacer(clip1,mode=0, tff=1)

deint  .set_output()

```
