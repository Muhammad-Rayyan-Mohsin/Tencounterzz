import Nav from '@/components/Nav'
import ResultsView from '@/components/ResultsView'

export default function HistoryRunPage({
  params,
}: {
  params: { runId: string }
}) {
  return (
    <>
      <Nav />
      <main className="min-h-[100dvh] pt-14">
        <ResultsView runId={params.runId} />
      </main>
    </>
  )
}
