#!/usr/bin/env perl
use strict;
use warnings;
use JSON;
use LWP::UserAgent;
use HTTP::Request;

# ---------------------------------------------------------------------------
# CanvasXpress MCP Client — Perl
# Install: cpan LWP::UserAgent JSON
#
# Usage:
#   perl test_client.pl
#   perl test_client.pl "Violin plot" "Gene,CellType,Expression"
#   perl test_client.pl "Heatmap" '{"Gene":"string","Sample1":"numeric","Treatment":"factor"}'
#   perl test_client.pl "Heatmap" "Gene,Sample1,Treatment" '{"Gene":"string","Sample1":"numeric","Treatment":"factor"}'
# ---------------------------------------------------------------------------

my $MCP_URL     = $ENV{MCP_URL} || "http://localhost:8100/mcp";
my $description = $ARGV[0]      || "Clustered heatmap with RdBu colors";

my $ua   = LWP::UserAgent->new(timeout => 120);
my $json = JSON->new->utf8->pretty;

# ── Parse arguments flexibly ─────────────────────────────────────────────────
my (%tool_args, $column_types, $headers, $data);

for my $arg (@ARGV[1..$#ARGV]) {
    if ($arg =~ /^\{/) {
        $column_types = decode_json($arg);
    } elsif ($arg =~ /^\[/) {
        $data = decode_json($arg);
    } else {
        $headers = [split /,/, $arg];
    }
}

# Default: sample data + types
unless ($headers || $data) {
    $data = [
        ["Gene",  "Sample1", "Sample2", "Treatment"],
        ["BRCA1", 1.2,       3.4,       "Control"  ],
        ["TP53",  2.1,       0.9,       "Treated"  ],
        ["EGFR",  0.8,       2.3,       "Control"  ],
    ];
    $column_types = {Gene => "string", Sample1 => "numeric", Sample2 => "numeric", Treatment => "factor"};
}

$tool_args{description}  = $description;
$tool_args{data}         = $data         if $data;
$tool_args{headers}      = $headers      if $headers;
$tool_args{column_types} = $column_types if $column_types;

# Print summary
print "Description : $description\n";
if ($data) {
    print "Data        : " . (scalar(@$data)-1) . " rows x " . scalar(@{$data->[0]}) . " columns\n";
    print "Columns     : " . join(", ", @{$data->[0]}) . "\n";
} elsif ($headers) {
    print "Headers     : " . join(", ", @$headers) . "\n";
}
if ($column_types) {
    print "Types       : " . join(", ", map { "$_=$column_types->{$_}" } sort keys %$column_types) . "\n";
}
print "\nConnecting to $MCP_URL...\n";

# Step 1: Initialize
my $init = post_mcp($ua, $MCP_URL, undef, {
    jsonrpc => "2.0", id => 1, method => "initialize",
    params  => {
        protocolVersion => "2024-11-05",
        capabilities    => {},
        clientInfo      => { name => "perl-client", version => "1.0.0" },
    },
});
die "Initialize failed: " . ($init->{error}{message} // "unknown") . "\n" if $init->{error};
my $session_id = $init->{_session_id};
print "Connected! Session: $session_id\n\n" if $session_id;

# Step 2: Notify initialized
post_mcp($ua, $MCP_URL, $session_id, {
    jsonrpc => "2.0", method => "notifications/initialized", params => {},
});

# Step 3: Call tool
my $result = post_mcp($ua, $MCP_URL, $session_id, {
    jsonrpc => "2.0", id => 2, method => "tools/call",
    params  => { name => "generate_canvasxpress_config", arguments => \%tool_args },
});
die "Tool call failed: " . ($result->{error}{message} // "unknown") . "\n" if $result->{error};

my $response = decode_json($result->{result}{content}[0]{text});

# Summary
if ($response->{headers_used} && @{$response->{headers_used}}) {
    print "Headers used : " . join(", ", @{$response->{headers_used}}) . "\n";
}
if ($response->{types_used} && %{$response->{types_used}}) {
    my %t = %{$response->{types_used}};
    print "Types used   : " . join(", ", map { "$_=$t{$_}" } sort keys %t) . "\n";
}
print "\n";

# Config
print "── Config ──────────────────────────────────\n";
print $json->encode($response->{config});

# Validation
print "\n── Validation ──────────────────────────────\n";
if ($response->{valid}) {
    print "All column references are valid\n";
} else {
    print "Column reference warnings:\n";
    print "   * $_\n" for @{$response->{warnings}};
    print "\n   Invalid refs: " . $json->encode($response->{invalid_refs}) . "\n" if %{$response->{invalid_refs}};
}

# ---------------------------------------------------------------------------
sub post_mcp {
    my ($ua, $url, $session_id, $payload) = @_;
    my $req = HTTP::Request->new(POST => $url);
    $req->header("Content-Type"   => "application/json");
    $req->header("Accept"         => "application/json, text/event-stream");
    $req->header("Mcp-Session-Id" => $session_id) if $session_id;
    $req->content($json->encode($payload));
    my $res  = $ua->request($req);
    my $body = $res->decoded_content // $res->content // "";
    my $ct   = $res->content_type // "";
    if ($ct =~ /event-stream/ || $body =~ /^event:/m || $body =~ /^data:/m) {
        my $js = _extract_sse_json($body) // return {};
        my $d  = eval { decode_json($js) }; warn "JSON error: $@\n" if $@; $d //= {};
        $d->{_session_id} = $res->header("Mcp-Session-Id") if $res->header("Mcp-Session-Id");
        return $d;
    }
    return {} unless $body && $body =~ /^\s*[\{\[]/;
    my $d = eval { decode_json($body) } // {};
    $d->{_session_id} = $res->header("Mcp-Session-Id") if $res->header("Mcp-Session-Id");
    return $d;
}

sub _extract_sse_json {
    my ($body) = @_;
    my (@chunks, $cur) = ((), "");
    for my $line (split /\n/, $body) {
        $line =~ s/\r$//;
        if ($line =~ /^data:\s*(.*)$/) { $cur .= $1 }
        elsif ($line eq "" && $cur ne "") { push @chunks, $cur; $cur = "" }
    }
    push @chunks, $cur if $cur ne "";
    for my $c (reverse @chunks) { $c =~ s/^\s+|\s+$//g; return $c if $c =~ /^\{/ }
    return undef;
}
