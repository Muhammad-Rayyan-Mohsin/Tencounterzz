import Nav from '@/components/Nav'
import ProcessingView from '@/components/ProcessingView'

export default function ProcessingPage({
  params,
}: {
  params: { jobId: string }
}) {
  return (
    <>
      <Nav />
      <main className="min-h-[100dvh] pt-14 flex items-center justify-center px-6 py-16">
        <div className="w-full max-w-2xl">
          <ProcessingView jobId={params.jobId} />
        </div>
      </main>
    </>
  )
}
