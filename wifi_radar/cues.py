#!/usr/bin/env python3
"""
Audible prompts, so a trial does not need eyes on the terminal.

An antenna trial is a person moving on cue and a wire that does not care when
they move. The cue is printed, but printing it asks the person to watch a screen
while holding a hand at 20 cm — and a trial run by an agent, or from a script,
has its output captured where nobody can see it at all. So the change of segment
is also announced: two tones and, where the machine can speak, the distance.

Nothing here may break a recording. A box with no speech engine, a locked-down
PowerShell, a terminal with its bell turned off: each of those ends in silence
and the printed line stays the record of what happened, which is the same
guarantee `--record` gives for frames.

Windows is the one platform with a tone in the standard library (`winsound`),
and its speech is the SAPI voice already installed. Elsewhere the bell is the
ASCII BEL and speech comes from `say` or `espeak` if one is there.

Speech is spoken by a process kept open for the session. Starting a speech
engine costs about three seconds on Windows — PowerShell and SAPI both load
before the first word — and a cue that arrives three seconds into a fifteen
second stretch is a cue the hand follows late, which quietly corrupts the marks
it is supposed to describe. One process reading one line per prompt answers in
milliseconds instead.
"""

import os
import shutil
import subprocess
import sys
import threading

#: Where the spoken text is handed over, so a sentence full of punctuation can
#: never be read as part of a shell command.
SPEECH_ENV = "WIFI_RADAR_CUE"

#: How long a speech command may take before it is abandoned. A missing voice
#: can block, and a cue that arrives after the segment has ended is worse than
#: no cue at all.
SPEECH_TIMEOUT = 10.0

MODES = ("off", "bell", "voice", "both")


def speech_command(text=None):
    """
    The argv that speaks `text` on this machine, or None.

    `text` is None for the "is there one?" question, which is asked once at
    startup so the terminal can say whether the voice will be heard.
    """
    if sys.platform == "win32":
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if not shell:
            return None
        return [shell, "-NoProfile", "-NonInteractive", "-Command",
                "Add-Type -AssemblyName System.Speech; "
                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                f".Speak($env:{SPEECH_ENV})"]
    if sys.platform == "darwin":
        command = shutil.which("say")
        return [command, text] if command and text is not None else (
            [command] if command else None)
    for name in ("spd-say", "espeak", "espeak-ng"):
        command = shutil.which(name)
        if command:
            return [command, text] if text is not None else [command]
    return None


def stdin_speech_command():
    """
    The argv of a speech engine that reads one prompt per line on stdin.

    `say` and `espeak --stdin` do this natively; PowerShell needs a loop, and
    the text is read from the pipe rather than interpolated into the script, so
    nothing in a prompt can become code.
    """
    if sys.platform == "win32":
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if not shell:
            return None
        return [shell, "-NoProfile", "-NonInteractive", "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "while ($true) { $line = [Console]::In.ReadLine(); "
                "if ($null -eq $line) { break }; $s.Speak($line) }"]
    if sys.platform == "darwin":
        command = shutil.which("say")
        return [command] if command else None
    for name in ("espeak", "espeak-ng"):
        command = shutil.which(name)
        if command:
            return [command, "--stdin"]
    return None


class Speaker:
    """
    A speech process kept open for the session.

    `say` is a write plus a flush: the engine is already loaded and the queued
    sentence is spoken as soon as the previous one ends. If the process cannot
    be started, or dies, `say` reports False and the cue falls back to whatever
    else it has.
    """

    def __init__(self, command=None):
        self._command = stdin_speech_command() if command is None else command
        self._process = None

    @property
    def available(self):
        return self._command is not None

    def start(self):
        if self._process is not None:
            return True
        if self._command is None:
            return False
        try:
            self._process = subprocess.Popen(
                self._command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                **({} if sys.platform != "win32"
                   else {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}))
        except (OSError, ValueError):
            self._process = None
        return self._process is not None

    def say(self, text):
        if not self.start() or not text:
            return False
        try:
            # One prompt per line, so a line break inside a prompt would be
            # read as two prompts and the second would arrive out of order.
            self._process.stdin.write(text.replace("\n", " ") + "\n")
            self._process.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def close(self, timeout=5.0):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()


def beep(times=2, frequency=880, duration=0.12):
    """
    Two tones, or the terminal bell.

    The tone says "the segment changed" without saying which — that is the
    sentence's job — and it is what still works when speech does not.
    """
    try:
        if sys.platform == "win32":
            import winsound
        else:
            winsound = None
    except Exception:
        winsound = None
    if winsound is not None:
        try:
            for _ in range(max(1, int(times))):
                winsound.Beep(int(frequency), int(duration * 1000))
            return True
        except Exception:
            pass
    try:
        sys.stdout.write("\a" * max(1, int(times)))
        sys.stdout.flush()
        return True
    except Exception:
        return False


def speak(text, timeout=SPEECH_TIMEOUT):
    """Say `text` out loud, or return False and say nothing at all."""
    command = speech_command(text)
    if command is None:
        return False
    environment = dict(os.environ)
    environment[SPEECH_ENV] = text
    try:
        return subprocess.run(command, env=environment, timeout=timeout,
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class Cue:
    """
    An announcement, made on a thread so the frames keep arriving.

    Speaking costs a few hundred milliseconds on Windows and a fraction of that
    elsewhere; doing it inline would drop a dozen frames per segment change, and
    the recording is the thing being protected. Threads are daemons: a cue still
    being spoken when the process ends is abandoned rather than waited for.
    """

    def __init__(self, mode="both", speak=None, beep=None, background=True,
                 say=None, speaker=None):
        if mode not in MODES:
            raise ValueError(f"unknown cue mode {mode!r}; one of {MODES}")
        self.mode = mode
        self._speak = speak or globals()["speak"]
        self._beep = beep or globals()["beep"]
        self._background = background
        self._say = say
        # A caller that injects `speak` wants its own function called, not a
        # process started behind its back.
        self._speaker = (speaker if speaker is not None else
                         (Speaker() if self.uses_voice and speak is None else None))
        self._threads = []
        self.spoken = 0
        self.beeped = 0

    @property
    def uses_voice(self):
        return self.mode in ("voice", "both")

    @property
    def uses_bell(self):
        return self.mode in ("bell", "both")

    def describe(self):
        """What this cue will actually do, for the first line of a trial."""
        if self.mode == "off":
            return "off — watch the printed lines"
        parts = []
        if self.uses_bell:
            parts.append("tones")
        if self.uses_voice:
            parts.append("speech" if stdin_speech_command() else
                         "speech unavailable, tones only")
        return " + ".join(parts) + f" at every segment change ({self.mode})"

    # ------------------------------------------------------------ lifecycle
    def start(self):
        """Open the speech engine now, so the first prompt is not late."""
        if self._speaker is not None:
            return self._speaker.start()
        return False

    def close(self, timeout=1.0):
        """Let queued prompts finish, then stop the engine."""
        for thread in list(self._threads):
            if thread.is_alive():
                thread.join(timeout)
        if self._speaker is not None:
            self._speaker.close()

    # ------------------------------------------------------------- making it
    def announce(self, text):
        """Cue the change to `text`. Never raises, never blocks the caller."""
        if self.mode == "off" or not text:
            return
        if not self._background:
            self._deliver(text)
            return
        try:
            thread = threading.Thread(target=self._deliver, args=(text,),
                                      daemon=True)
            thread.start()
            self._threads.append(thread)
        except Exception:
            # A thread that cannot start is a machine out of resources; the
            # trial is more important than the cue.
            pass

    def _deliver(self, text):
        try:
            if self.uses_bell and self._beep():
                self.beeped += 1
            spoke = False
            if self.uses_voice:
                if self._speaker is not None:
                    spoke = bool(self._speaker.say(text))
                    if not spoke:
                        # The engine did not take it; one short-lived process is
                        # still better than silence.
                        spoke = bool(self._speak(text))
                else:
                    spoke = bool(self._speak(text))
                if spoke:
                    self.spoken += 1
                elif self.uses_bell:
                    # The voice was the point and did not arrive: two more
                    # tones at least say "now", which is all a segment change
                    # needs to say.
                    self._beep()
            if self._say:
                self._say(text)
        except Exception:
            # Any failure here is a failed cue, not a failed trial.
            pass
