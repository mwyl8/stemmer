import { useEffect } from 'react'

const EDITABLE_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT'])

function isTypingTarget(el) {
  if (!el) return false
  if (EDITABLE_TAGS.has(el.tagName)) return true
  if (el.isContentEditable) return true
  return false
}

/**
 * Global transport shortcuts (PRD §2.1): space (play/pause), ←/→ (skip),
 * M (mute focused stem), S (solo focused stem), 0-9 (jump to 0%-90% of
 * track). Ignored while focus is in a text input/textarea/select/
 * contenteditable so typing a URL or filename never triggers playback
 * changes. `player` is the object returned by useMultitrackPlayer; pass
 * `null`/`undefined` (e.g. before the player is ready) to disable.
 */
export default function useKeyboardShortcuts(player) {
  useEffect(() => {
    if (!player) return

    function handleKeyDown(event) {
      if (isTypingTarget(document.activeElement)) return
      if (event.metaKey || event.ctrlKey || event.altKey) return

      switch (event.key) {
        case ' ':
        case 'Spacebar':
          event.preventDefault()
          player.togglePlay()
          break
        case 'ArrowLeft':
          event.preventDefault()
          player.skip(event.shiftKey ? -30 : -10)
          break
        case 'ArrowRight':
          event.preventDefault()
          player.skip(event.shiftKey ? 30 : 10)
          break
        case 'm':
        case 'M':
          if (player.focusedIndex != null) player.toggleMute(player.focusedIndex)
          break
        case 's':
        case 'S':
          if (player.focusedIndex != null) player.toggleSolo(player.focusedIndex)
          break
        default:
          if (event.key >= '0' && event.key <= '9') {
            player.jumpToPercent(Number(event.key) * 10)
          }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [player])
}
