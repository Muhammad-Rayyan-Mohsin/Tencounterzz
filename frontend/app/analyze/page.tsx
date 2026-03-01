import Nav from '@/components/Nav'
import VideoDropzone from '@/components/VideoDropzone'

export default function AnalysePage() {
  return (
    <>
      <Nav />
      <main className="min-h-[100dvh] pt-14">
        <div className="max-w-[1400px] mx-auto grid grid-cols-1 lg:grid-cols-[1fr_1.15fr] min-h-[calc(100dvh-3.5rem)]">

          {/* Left — Identity panel */}
          <div className="flex flex-col justify-center px-8 py-16 lg:px-16 lg:py-24 border-r border-white/[0.05]">
            <div className="flex items-center gap-2 mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
              <span className="text-xs font-mono uppercase tracking-widest text-zinc-500">
                FYP — Boxing Analytics v2
              </span>
            </div>

            <h1 className="text-5xl lg:text-[4.25rem] font-semibold tracking-tighter leading-[0.95] text-zinc-50 text-balance mb-6">
              Fight footage,
              <br />
              <span className="text-zinc-500">broken down</span>
              <br />
              to the punch.
            </h1>

            <p className="text-zinc-400 text-base leading-relaxed max-w-[52ch] mb-10">
              Upload a boxing video and the full two-stage pipeline runs
              automatically — YOLOv11m detection, YOLOv8m-pose estimation, and
              AttentionBiLSTM punch classification across 6 punch types.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[
                { label: 'Person Detection', sub: 'YOLOv11m — 2 fighter tracking' },
                { label: 'Pose Estimation', sub: '17 COCO keypoints per frame' },
                { label: 'Punch Classification', sub: 'AttentionBiLSTM, 6 classes' },
                { label: 'Impact Analytics', sub: 'Elbow angle + wrist velocity' },
              ].map((item) => (
                <div
                  key={item.label}
                  className="flex items-start gap-3 p-3 rounded-xl border border-white/[0.05] bg-white/[0.02]"
                >
                  <span className="mt-0.5 w-1 h-full min-h-[2rem] rounded-full bg-rose-600/60 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-zinc-200">{item.label}</p>
                    <p className="text-xs text-zinc-500 mt-0.5">{item.sub}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-10 pt-8 border-t border-white/[0.05] grid grid-cols-3 gap-4">
              {[
                { value: '70.9%', label: 'Classifier accuracy' },
                { value: '6', label: 'Punch classes' },
                { value: '30fps', label: 'Output video' },
              ].map((s) => (
                <div key={s.label}>
                  <p className="text-xl font-semibold font-mono tracking-tight text-zinc-100">
                    {s.value}
                  </p>
                  <p className="text-xs text-zinc-500 mt-0.5">{s.label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right — Upload panel */}
          <div className="flex items-center justify-center px-8 py-16 lg:px-14 lg:py-24 bg-grid-pattern bg-grid-sm">
            <div className="w-full max-w-[540px]">
              <VideoDropzone />
            </div>
          </div>
        </div>
      </main>
    </>
  )
}
