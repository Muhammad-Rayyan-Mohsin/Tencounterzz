import { NextResponse } from 'next/server'
import { stat, readFile } from 'fs/promises'
import path from 'path'

const UPLOADS = path.join(process.cwd(), 'public', 'uploads')

export async function GET(
  _request: Request,
  { params }: { params: { path: string[] } }
) {
  const filename = params.path.join('/')

  // Prevent directory traversal
  if (filename.includes('..')) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 })
  }

  const filePath = path.join(UPLOADS, filename)

  try {
    const fileStat = await stat(filePath)
    if (!fileStat.isFile()) {
      return NextResponse.json({ error: 'Not found' }, { status: 404 })
    }

    const data = await readFile(filePath)
    const ext = path.extname(filename).toLowerCase()
    const contentType =
      ext === '.mp4' ? 'video/mp4' :
      ext === '.webm' ? 'video/webm' :
      ext === '.png' ? 'image/png' :
      ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' :
      'application/octet-stream'

    return new NextResponse(data, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': fileStat.size.toString(),
        'Accept-Ranges': 'bytes',
      },
    })
  } catch {
    return NextResponse.json({ error: 'Not found' }, { status: 404 })
  }
}
