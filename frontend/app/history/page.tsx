import Nav from '@/components/Nav'
import HistoryList from '@/components/HistoryList'

export const dynamic = 'force-dynamic'

export default function HistoryPage() {
  return (
    <>
      <Nav />
      <main className="min-h-[100dvh] pt-14">
        <HistoryList />
      </main>
    </>
  )
}
