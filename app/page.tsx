"use client"
import React, { useEffect, useRef, useState } from "react"

type Message = {
  role: "user" | "assistant"
  content: string
}

type UploadedFile = {
  name: string
  size: number
  type: string
  key: string
  url: string
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [uploading, setUploading] = useState(false)
  const endRef = useRef<HTMLDivElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function uploadFile(file: File) {
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append("file", file)

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const errorData = await res.json()
        throw new Error(errorData.error || "Upload failed")
      }

      const data = await res.json()
      setUploadedFiles((prev) => [...prev, data.file])

      // Add a system message about the upload
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `✓ File uploaded: ${data.file.name} (${(data.file.size / 1024).toFixed(2)} KB)`,
        },
      ])
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error"
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Upload error: ${errorMessage}`,
        },
      ])
    } finally {
      setUploading(false)
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    await uploadFile(file)

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  async function handleSampleUpload() {
    try {
      // Fetch fresh presigned URL from backend
      const urlRes = await fetch("/api/sample-pdf");
      if (!urlRes.ok) {
        const errorData = await urlRes.json();
        throw new Error(errorData.error || "Failed to get sample PDF");
      }

      const { url, fileName } = await urlRes.json();

      // Download the PDF from the presigned URL
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error("Failed to download sample PDF");
      }

      const blob = await res.blob();
      const sampleFile = new File([blob], fileName, {
        type: blob.type || "application/pdf",
      });

      await uploadFile(sampleFile);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Upload error: ${errorMessage}`,
        },
      ]);
    }
  }

  function removeFile(index: number) {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  async function sendMessage(e?: React.FormEvent) {
    e?.preventDefault()
    const text = input.trim()
    if (!text || loading) return
    setLoading(true)

    const newUserMsg: Message = { role: "user", content: text }
    const history = [...messages, newUserMsg]
    setMessages(history)
    setInput("")

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: text, 
          history,
          files: uploadedFiles // Include uploaded files in the request
        }),
      })

      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`)
      }

      const contentType = res.headers.get("content-type") || ""
      let replyText = ""

      if (contentType.includes("text/event-stream")) {
        // Handle streaming response
        const reader = res.body?.getReader()
        const decoder = new TextDecoder()

        if (!reader) {
          throw new Error("No response body")
        }

        // Add an empty assistant message to append to
        setMessages((prev) => [...prev, { role: "assistant", content: "" }])

        let buffer = ""
        let replyText = ""
        let streamFinished = false

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              if (!streamFinished) {
                console.warn("Stream ended prematurely");
              }
              break
            }

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split("\n")
            
            // Keep the last incomplete line in the buffer
            buffer = lines[lines.length - 1]

            for (let i = 0; i < lines.length - 1; i++) {
              const line = lines[i].trim()
              if (line.startsWith("data: ")) {
                try {
                  const data = JSON.parse(line.slice(6))
                  if (data.chunk) {
                    replyText += data.chunk
                    // Update the last message with accumulated text in real-time
                    setMessages((prev) => {
                      const updated = [...prev]
                      if (updated[updated.length - 1].role === "assistant") {
                        updated[updated.length - 1].content = replyText
                      }
                      return updated
                    })
                  } else if (data.done) {
                    // Stream finished successfully
                    streamFinished = true
                    break
                  } else if (data.error) {
                    // Stream error
                    throw new Error(data.error)
                  }
                } catch (e) {
                  if (e instanceof SyntaxError) {
                    console.error("Error parsing stream data:", e)
                  } else {
                    throw e
                  }
                }
              }
            }

            if (streamFinished) break
          }
        } catch (streamError) {
          const errorMsg = streamError instanceof Error ? streamError.message : "Stream error"
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `⚠️ ${errorMsg}` },
          ])
        }
      } else if (contentType.includes("application/json")) {
        const data = await res.json()
        replyText =
          data?.reply || data?.answer || data?.output || data?.text || "(no reply)"
        setMessages((prev) => [...prev, { role: "assistant", content: replyText }])
      } else {
        replyText = await res.text()
        setMessages((prev) => [...prev, { role: "assistant", content: replyText }])
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "unknown error"
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errorMessage}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 dark:bg-black">
      <header className="border-b border-zinc-200 dark:border-zinc-800 px-6 py-4">
        <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">RAG Chat Tester</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">Simple UI to try your API</p>
      </header>

      <main className="flex-1 mx-auto w-full max-w-3xl px-6 py-6">
        {/* Uploaded Files Display */}
        {uploadedFiles.length > 0 && (
          <div className="mb-4 space-y-2">
            <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Uploaded Files:
            </h3>
            <div className="space-y-2">
              {uploadedFiles.map((file, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-4 py-2"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">📄</span>
                    <div>
                      <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        {file.name}
                      </p>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        {(file.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => removeFile(i)}
                    className="text-sm text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {messages.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-300 dark:border-zinc-700 p-8 text-center text-zinc-600 dark:text-zinc-400">
            Start by typing a message below.
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((m, i) => (
              <div
                key={i}
                className={
                  m.role === "user"
                    ? "flex justify-end"
                    : "flex justify-start"
                }
              >
                <div
                  className={
                    m.role === "user"
                      ? "max-w-[80%] rounded-2xl bg-zinc-900 text-zinc-50 px-4 py-3"
                      : "max-w-[80%] rounded-2xl bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 border border-zinc-200 dark:border-zinc-800 px-4 py-3"
                  }
                >
                  <div className="text-xs mb-1 opacity-60">
                    {m.role === "user" ? "You" : "Assistant"}
                  </div>
                  <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
                </div>
              </div>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </main>

      <form
        className="sticky bottom-0 w-full border-t border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-black/80 backdrop-blur supports-backdrop-filter:bg-white/50 supports-backdrop-filter:dark:bg-black/50"
        onSubmit={sendMessage}
      >
        <div className="mx-auto w-full max-w-3xl px-6 py-4">
          <div className="flex gap-2">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".pdf,.doc,.docx,.txt"
              className="hidden"
            />
            <button
              type="button"
              onClick={handleSampleUpload}
              disabled={uploading}
              className="rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-700 text-zinc-900 dark:text-zinc-100 px-4 py-3 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-zinc-50 dark:hover:bg-zinc-800"
              title="Load sample PDF"
            >
              {uploading ? "📤..." : "📎 Sample"}
            </button>
            <input
              aria-label="Message"
              placeholder="Ask something..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              className="flex-1 rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 px-4 py-3 text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:focus:ring-zinc-600"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="rounded-xl bg-zinc-900 text-zinc-50 px-5 py-3 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-zinc-800"
            >
              {loading ? "Sending…" : "Send"}
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}
