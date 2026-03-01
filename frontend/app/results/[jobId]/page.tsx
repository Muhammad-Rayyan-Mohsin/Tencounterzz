import Nav from '@/components/Nav'
import ResultsView from '@/components/ResultsView'

export default function ResultsPage({
  params,
}: {
  params: { jobId: string }
}) {
  return (
    <>
      <Nav />
      <main className="min-h-[100dvh] pt-14">
        <ResultsView jobId={params.jobId} />
      </main>
    </>
  )
}
