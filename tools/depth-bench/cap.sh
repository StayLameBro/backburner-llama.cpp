#!/usr/bin/env bash
# cap.sh SECONDS cmd... : run cmd in its own process group; after SECONDS send TERM then KILL to the
# whole group (server included) and exit 124. macOS has no coreutils timeout.
secs=$1; shift
exec perl -e '
  my $s = shift @ARGV; my $pid = fork();
  if (!$pid) { setpgrp(0, 0); exec @ARGV or die "exec: $!"; }
  $SIG{ALRM} = sub { print STDERR "cap: killed after ${s}s\n"; kill "-TERM", $pid; sleep 3; kill "-KILL", $pid; exit 124 };
  alarm $s; waitpid($pid, 0); exit($? >> 8);' "$secs" "$@"
