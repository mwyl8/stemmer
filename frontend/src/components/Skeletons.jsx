/** Loading placeholders (PRD §2 polish: "loading skeletons ... proper
 * empty/error states") — shown while the very first GET /jobs/{id} is in
 * flight, before there's any real job data to render a shape from. */
export function JobPageSkeleton() {
  return (
    <div className="mx-auto flex max-w-xl flex-col gap-6 px-4 py-24">
      <div className="flex flex-col gap-2">
        <div className="skeleton h-6 w-64" />
        <div className="skeleton h-4 w-40" />
      </div>
      <div className="flex flex-col gap-3">
        <div className="flex gap-2">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="flex-1">
              <div className="skeleton h-2 w-full" />
              <div className="skeleton mt-1.5 h-3 w-12" />
            </div>
          ))}
        </div>
        <div className="skeleton h-4 w-48" />
      </div>
    </div>
  )
}

export function ResultViewSkeleton() {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 px-4 py-12">
      <div className="skeleton h-6 w-40" />
      <div className="skeleton h-12 w-full rounded-xl" />
      <div className="flex flex-col gap-2">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="skeleton h-14 w-full rounded-lg" />
        ))}
      </div>
      <div className="skeleton h-32 w-full rounded-xl" />
    </div>
  )
}
