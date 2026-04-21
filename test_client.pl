#!/usr/bin/env perl
use strict;
use warnings;
use JSON;
use LWP::UserAgent;
use HTTP::Request;
use URI::Escape qw(uri_escape);

# ---------------------------------------------------------------------------
# CanvasXpress MCP Test Client — Perl
# Install: cpan LWP::UserAgent JSON
#
# Usage:
#   # Run built-in examples (generate + modify)
#   perl test_client.pl --examples
#
#   # Generate a config
#   perl test_client.pl "Violin plot by cell type" "Gene,CellType,Expression"
#   perl test_client.pl "Heatmap" '{"Gene":"string","Sample1":"numeric","Treatment":"factor"}'
#   perl test_client.pl "Scatter plot" "Gene,Expr,Treatment" '{"Gene":"string","Expr":"numeric","Treatment":"factor"}'
#
#   # Modify an existing config
#   perl test_client.pl --modify '{"graphType":"Bar","xAxis":["Gene"]}' "add a title My Chart"
#   perl test_client.pl --modify '{"graphType":"Heatmap","xAxis":["Gene"]}' "change colorScheme to Spectral"
#
#   # Create a map config (via MCP tool)
#   perl test_client.pl --map World
#   perl test_client.pl --map USAStates --title "Sales by State" --color-scheme Blues
#   perl test_client.pl --map USA
#   perl test_client.pl --map Europe --title "EU Revenue"
#   perl test_client.pl --map CA --title "California Counties"
#
#   # REST endpoint (no MCP protocol — plain HTTP GET)
#   perl test_client.pl --rest generate "Violin plot" "Gene,Expr,CellType" '{"Gene":"string","Expr":"numeric","CellType":"factor"}'
#   perl test_client.pl --rest modify '{"graphType":"Bar","xAxis":["Gene"]}' "add a title My Chart"
#   perl test_client.pl --rest map World
#   perl test_client.pl --rest map USAStates --title "Sales by State" --color-scheme Blues
#   perl test_client.pl --rest generate "Bar chart" --target myCanvas           # echoes target in response
#   perl test_client.pl --rest generate "Bar chart" --callback renderChart      # JSONP response
#   perl test_client.pl --rest generate "Bar chart" --client-id req-1 --callback renderChart
# ---------------------------------------------------------------------------

my $MCP_URL  = $ENV{MCP_URL}  || "http://localhost:8100/mcp";
my $REST_URL = $ENV{REST_URL} || "http://localhost:8100";
my $ua      = LWP::UserAgent->new(timeout => 120);
my $json    = JSON->new->utf8->pretty;
my $SEP     = "─" x 50;
my $SEP2    = "═" x 50;

# ---------------------------------------------------------------------------
# Built-in examples
# ---------------------------------------------------------------------------

my @GENERATE_EXAMPLES = (
    {
        label       => "Clustered heatmap",
        description => "Clustered heatmap with RdBu colors and dendrograms on both axes",
        data        => [
            ["Gene",  "Control1", "Control2", "Drug1", "Drug2"],
            ["BRCA1", 2.1,        0.9,        3.8,     3.2   ],
            ["TP53",  1.2,        1.4,        0.3,     0.5   ],
            ["EGFR",  0.8,        0.6,        2.9,     3.1   ],
            ["MYC",   3.2,        2.8,        0.4,     0.6   ],
        ],
        column_types => {Gene=>"string", Control1=>"numeric", Control2=>"numeric",
                         Drug1=>"numeric", Drug2=>"numeric"},
    },
    {
        label       => "Volcano plot",
        description => "Volcano plot with log2 fold change on x-axis and -log10 p-value on y-axis",
        data        => [
            ["Gene",  "log2FC", "negLog10P"],
            ["GeneA",  2.3,      4.1       ],
            ["GeneB", -1.8,      3.7       ],
            ["GeneC",  0.2,      0.4       ],
            ["GeneD",  3.1,      6.2       ],
        ],
        column_types => {Gene=>"string", log2FC=>"numeric", negLog10P=>"numeric"},
    },
    {
        label        => "Violin plot",
        description  => "Violin plot of gene expression grouped by cell type with Tableau colors",
        headers      => ["CellID", "Expression", "CellType"],
        column_types => {CellID=>"string", Expression=>"numeric", CellType=>"factor"},
    },
    {
        label        => "PCA scatter plot",
        description  => "PCA scatter plot with PC1 vs PC2 colored by Treatment with regression ellipses",
        headers      => ["Sample", "PC1", "PC2", "Treatment"],
        column_types => {Sample=>"string", PC1=>"numeric", PC2=>"numeric", Treatment=>"factor"},
    },
    {
        label        => "Kaplan-Meier survival curve",
        description  => "Kaplan-Meier survival curve for two treatment groups",
        headers      => ["Patient", "Time", "Event", "Treatment"],
        column_types => {Patient=>"string", Time=>"numeric", Event=>"numeric", Treatment=>"factor"},
    },
    {
        label       => "Stacked percent bar",
        description => "Stacked percent bar chart of market share by year and company",
        data        => [
            ["Company", "Y2021", "Y2022", "Y2023"],
            ["Alpha",    35,      28,      31     ],
            ["Beta",     28,      33,      29     ],
            ["Gamma",    37,      39,      40     ],
        ],
        column_types => {Company=>"string", Y2021=>"numeric", Y2022=>"numeric", Y2023=>"numeric"},
    },
    {
        label        => "Ridgeline density",
        description  => "Ridgeline density plot of expression values by cell population",
        headers      => ["Cell", "Value", "Population"],
        column_types => {Cell=>"string", Value=>"numeric", Population=>"factor"},
    },
    {
        label        => "Sankey flow diagram",
        description  => "Sankey diagram showing patient flow from diagnosis through treatment to outcome",
        headers      => ["Diagnosis", "Treatment", "Outcome"],
        column_types => {Diagnosis=>"factor", Treatment=>"factor", Outcome=>"factor"},
    },
);

my @MAP_EXAMPLES = (
    { label => "World map",        map_id => "World",          title => "Global Overview",   color_scheme => "Blues"    },
    { label => "World continents", map_id => "WorldContinents", title => "World Continents"                             },
    {
        label => "USA States", map_id => "USAStates", title => "Sales by State", color_scheme => "YlOrRd",
        data  => [["State","Value"],["CA",120],["NY",98],["TX",87],["FL",74],["WA",55]],
    },
    { label => "USA Counties",       map_id => "USACounties", title => "County-level Data"                           },
    { label => "Country code (USA)", map_id => "USA",         title => "United States",      color_scheme => "Greens"  },
    { label => "Country code (CAN)", map_id => "CAN",         title => "Canada"                                       },
    { label => "Country name",       map_id => "Mexico",      title => "Mexico",             color_scheme => "Oranges" },
    { label => "Continent (Europe)", map_id => "Europe",      title => "European Map",       color_scheme => "Tableau" },
    { label => "US State code (CA)", map_id => "CA",          title => "California"                                   },
    { label => "US State name",      map_id => "Texas",       title => "Texas",              color_scheme => "Reds"    },
);

my @MODIFY_EXAMPLES = (
    {
        label        => "Add title and switch theme",
        start_config => {graphType=>"Heatmap", xAxis=>["Gene"],
                         samplesClustered=>JSON::true, variablesClustered=>JSON::true,
                         colorScheme=>"RdBu"},
        instruction  => "add a title Expression Heatmap and switch to dark theme",
    },
    {
        label        => "Change color scheme and add title",
        start_config => {graphType=>"Bar", xAxis=>["Region"], graphOrientation=>"horizontal"},
        instruction  => "change the color scheme to Tableau and add a title Regional Sales",
    },
    {
        label        => "Remove legend and set axis titles",
        start_config => {graphType=>"Scatter2D", xAxis=>["PC1"], yAxis=>["PC2"],
                         colorBy=>"Treatment", showLegend=>JSON::true},
        instruction  => "remove the legend and set xAxisTitle to PC1 (32%) and yAxisTitle to PC2 (18%)",
    },
    {
        label        => "Add grouping and jitter",
        start_config => {graphType=>"Boxplot", xAxis=>["Expression"]},
        instruction  => "add groupingFactors for the CellType column and enable jitter on the data points",
    },
);

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

my $first_arg = $ARGV[0] // "";

if ($first_arg eq "--examples") {
    run_examples();
    exit 0;
}

if ($first_arg eq "--rest") {
    # -----------------------------------------------------------------------
    # REST mode — calls /generate, /modify, or /map directly (no MCP protocol)
    # Supports ?callback=name (JSONP) and ?target=id
    # -----------------------------------------------------------------------
    my $sub_cmd = $ARGV[1] // die "Usage: --rest generate|modify|map ...\n";
    my %extras;

    # Scan for --callback / --target / --client-id flags anywhere in remaining args
    my @rest_args;
    my $i = 2;
    while ($i <= $#ARGV) {
        if    ($ARGV[$i] eq '--callback'  && defined $ARGV[$i+1]) { $extras{callback}  = $ARGV[++$i] }
        elsif ($ARGV[$i] eq '--target'    && defined $ARGV[$i+1]) { $extras{target}    = $ARGV[++$i] }
        elsif ($ARGV[$i] eq '--client-id' && defined $ARGV[$i+1]) { $extras{client_id} = $ARGV[++$i] }
        else { push @rest_args, $ARGV[$i] }
        $i++;
    }

    if ($sub_cmd eq 'generate') {
        my $description = $rest_args[0] // die "Missing description\n";
        my ($headers, $data, $column_types) = parse_extra_args(@rest_args[1..$#rest_args]);
        my %params = (description => $description);
        $params{headers}      = join(",", @$headers)          if $headers;
        $params{data}         = $json->encode($data)          if $data;
        $params{column_types} = encode_json($column_types)    if $column_types;
        $params{callback}     = $extras{callback}             if $extras{callback};
        $params{target}       = $extras{target}               if $extras{target};
        $params{client_id}    = $extras{client_id}            if $extras{client_id};
        my $url = "$REST_URL/generate?" . join("&", map { uri_escape($_) . "=" . uri_escape($params{$_}) } keys %params);
        print "GET $url\n\n";
        my $res  = $ua->get($url);
        my $body = $res->decoded_content;
        if ($extras{callback}) {
            print "── JSONP response $SEP\n$body\n";
        } else {
            my $r = decode_json($body);
            print_generate_result($r);
        }

    } elsif ($sub_cmd eq 'modify') {
        my $config      = decode_json($rest_args[0] // die "Missing config JSON\n");
        my $instruction = $rest_args[1]              // die "Missing instruction\n";
        my ($headers, $data, $column_types) = parse_extra_args(@rest_args[2..$#rest_args]);
        my %params = (config => encode_json($config), instruction => $instruction);
        $params{headers}      = join(",", @$headers)          if $headers;
        $params{data}         = $json->encode($data)          if $data;
        $params{column_types} = encode_json($column_types)    if $column_types;
        $params{callback}     = $extras{callback}             if $extras{callback};
        $params{target}       = $extras{target}               if $extras{target};
        $params{client_id}    = $extras{client_id}            if $extras{client_id};
        my $url = "$REST_URL/modify?" . join("&", map { uri_escape($_) . "=" . uri_escape($params{$_}) } keys %params);
        print "GET $url\n\n";
        my $res  = $ua->get($url);
        my $body = $res->decoded_content;
        if ($extras{callback}) {
            print "── JSONP response $SEP\n$body\n";
        } else {
            my $r = decode_json($body);
            print_modify_result($config, $r, $instruction);
        }

    } elsif ($sub_cmd eq 'map') {
        my $map_id = $rest_args[0] // die "Missing map_id\n";
        my %params = (map_id => $map_id);
        # Scan rest_args for --title / --color-scheme flags
        my $j = 1;
        while ($j <= $#rest_args) {
            if    ($rest_args[$j] eq '--title'        && defined $rest_args[$j+1]) { $params{title}        = $rest_args[++$j] }
            elsif ($rest_args[$j] eq '--color-scheme' && defined $rest_args[$j+1]) { $params{color_scheme} = $rest_args[++$j] }
            elsif ($rest_args[$j] =~ /^\[/)                                        { $params{data}         = $rest_args[$j]   }
            $j++;
        }
        $params{callback}  = $extras{callback}  if $extras{callback};
        $params{target}    = $extras{target}    if $extras{target};
        $params{client_id} = $extras{client_id} if $extras{client_id};
        my $url = "$REST_URL/map?" . join("&", map { uri_escape($_) . "=" . uri_escape($params{$_}) } keys %params);
        print "GET $url\n\n";
        my $res  = $ua->get($url);
        my $body = $res->decoded_content;
        if ($extras{callback}) {
            print "── JSONP response $SEP\n$body\n";
        } else {
            my $r = decode_json($body);
            print_map_result($r);
        }

    } else {
        die "Unknown sub-command '$sub_cmd'. Use: generate | modify | map\n";
    }
    exit 0;
}

if ($first_arg eq '--map') {
    my $map_id = $ARGV[1] // die "Usage: perl test_client.pl --map <map_id> [--title <title>] [--color-scheme <scheme>]\n";
    my ($title, $color_scheme, $data);
    my $i = 2;
    while ($i <= $#ARGV) {
        if    ($ARGV[$i] eq '--title'        && defined $ARGV[$i+1]) { $title        = $ARGV[++$i] }
        elsif ($ARGV[$i] eq '--color-scheme' && defined $ARGV[$i+1]) { $color_scheme = $ARGV[++$i] }
        elsif ($ARGV[$i] =~ /^\[/)                                   { $data         = decode_json($ARGV[$i]) }
        $i++;
    }
    print "Tool         : create_map_config\n";
    print "Map ID       : $map_id\n";
    print "Title        : $title\n"        if $title;
    print "Color scheme : $color_scheme\n" if $color_scheme;
    printf "Data rows    : %d\n", scalar(@$data)-1 if $data;
    print "\n";
    my $args = {map_id => $map_id};
    $args->{title}        = $title        if $title;
    $args->{color_scheme} = $color_scheme if $color_scheme;
    $args->{data}         = $data         if $data;
    my $response = call_tool("create_map_config", $args);
    print_map_result($response);
    exit 0;
}

if ($first_arg eq 'modify') {
    my $orig_config = decode_json($ARGV[1] // die "Missing config JSON\n");
    my $instruction = $ARGV[2]             // die "Missing instruction\n";
    my ($headers, $data, $column_types) = parse_extra_args(@ARGV[3..$#ARGV]);
    print "Tool        : modify_canvasxpress_config\n";
    print "Instruction : $instruction\n";
    print "Config keys : " . join(", ", sort keys %$orig_config) . "\n\n";
    my $response = call_tool("modify_canvasxpress_config", {
        config      => $orig_config,
        instruction => $instruction,
        ($data        ? (data         => $data)        : ()),
        ($headers     ? (headers      => $headers)     : ()),
        ($column_types? (column_types => $column_types): ()),
    });
    print_modify_result($orig_config, $response, $instruction);
    exit 0;
}

# Default: generate
my $description = $first_arg || "Clustered heatmap with RdBu colors";
my ($headers, $data, $column_types) = parse_extra_args(@ARGV[1..$#ARGV]);

unless ($headers || $data) {
    $data = [
        ["Gene",  "Sample1", "Sample2", "Treatment"],
        ["BRCA1", 1.2,       3.4,       "Control"  ],
        ["TP53",  2.1,       0.9,       "Treated"  ],
        ["EGFR",  0.8,       2.3,       "Control"  ],
    ];
    $column_types = {Gene=>"string", Sample1=>"numeric", Sample2=>"numeric", Treatment=>"factor"};
}

print "Tool        : generate_canvasxpress_config\n";
print "Description : $description\n";
if ($data) {
    printf "Data        : %d rows x %d columns  (%s)\n",
        scalar(@$data)-1, scalar(@{$data->[0]}), join(", ", @{$data->[0]});
} elsif ($headers) {
    print "Headers     : " . join(", ", @$headers) . "\n";
}
if ($column_types) {
    print "Types       : " . join(", ", map { "$_=$column_types->{$_}" } sort keys %$column_types) . "\n";
}
print "\n";

my $response = call_tool("generate_canvasxpress_config", {
    description => $description,
    ($data        ? (data         => $data)        : ()),
    ($headers     ? (headers      => $headers)     : ()),
    ($column_types? (column_types => $column_types): ()),
});
print_generate_result($response);

# ---------------------------------------------------------------------------
# Examples runner
# ---------------------------------------------------------------------------

sub run_examples {
    print "\n$SEP2\n  CanvasXpress MCP — Built-in Examples\n  Server : $MCP_URL\n$SEP2\n";

    print "\n$SEP\n  GENERATE EXAMPLES\n$SEP\n";
    my $n = scalar @GENERATE_EXAMPLES;
    for my $i (0..$#GENERATE_EXAMPLES) {
        my $ex = $GENERATE_EXAMPLES[$i];
        printf "\n[%d/%d] %s\n", $i+1, $n, $ex->{label};
        print "  Description : $ex->{description}\n";
        if ($ex->{data}) {
            my @rows = @{$ex->{data}};
            printf "  Data        : %d rows x %d columns  (%s)\n",
                scalar(@rows)-1, scalar(@{$rows[0]}), join(", ", @{$rows[0]});
        } elsif ($ex->{headers}) {
            print "  Headers     : " . join(", ", @{$ex->{headers}}) . "\n";
        }
        print "\n";
        eval {
            my $args = {description => $ex->{description}};
            $args->{data}         = $ex->{data}         if $ex->{data};
            $args->{headers}      = $ex->{headers}       if $ex->{headers};
            $args->{column_types} = $ex->{column_types}  if $ex->{column_types};
            my $resp = call_tool("generate_canvasxpress_config", $args);
            print_generate_result($resp);
        };
        print "  Error: $@\n" if $@;
        print "\n$SEP\n" if $i < $#GENERATE_EXAMPLES;
    }

    print "\n\n$SEP\n  MODIFY EXAMPLES\n$SEP\n";
    my $m = scalar @MODIFY_EXAMPLES;
    for my $i (0..$#MODIFY_EXAMPLES) {
        my $ex = $MODIFY_EXAMPLES[$i];
        printf "\n[%d/%d] %s\n", $i+1, $m, $ex->{label};
        print "  Instruction  : $ex->{instruction}\n";
        print "  Start config : " . JSON->new->utf8->encode($ex->{start_config}) . "\n\n";
        eval {
            my $resp = call_tool("modify_canvasxpress_config", {
                config      => $ex->{start_config},
                instruction => $ex->{instruction},
            });
            print_modify_result($ex->{start_config}, $resp, $ex->{instruction});
        };
        print "  Error: $@\n" if $@;
        print "\n$SEP\n" if $i < $#MODIFY_EXAMPLES;
    }

    print "\n\n$SEP\n  MAP EXAMPLES\n$SEP\n";
    my $nm = scalar @MAP_EXAMPLES;
    for my $i (0..$#MAP_EXAMPLES) {
        my $ex = $MAP_EXAMPLES[$i];
        printf "\n[%d/%d] %s\n", $i+1, $nm, $ex->{label};
        print "  Map ID      : $ex->{map_id}\n";
        print "  Title       : $ex->{title}\n"        if $ex->{title};
        print "  ColorScheme : $ex->{color_scheme}\n" if $ex->{color_scheme};
        printf "  Data        : %d rows\n", scalar(@{$ex->{data}})-1 if $ex->{data};
        print "\n";
        eval {
            my $args = {map_id => $ex->{map_id}};
            $args->{title}        = $ex->{title}        if $ex->{title};
            $args->{color_scheme} = $ex->{color_scheme} if $ex->{color_scheme};
            $args->{data}         = $ex->{data}         if $ex->{data};
            my $resp = call_tool("create_map_config", $args);
            print_map_result($resp);
        };
        print "  Error: $@\n" if $@;
        print "\n$SEP\n" if $i < $#MAP_EXAMPLES;
    }

    print "\n$SEP2\n\n";
}

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

sub print_map_result {
    my ($response) = @_;
    print "Map ID       : " . ($response->{map_id} // "?") . "\n";
    if ($response->{headers_used} && @{$response->{headers_used}}) {
        print "Data columns : " . join(", ", @{$response->{headers_used}}) . "\n";
    }
    print "\n── Config $SEP\n";
    print $json->encode($response->{config});
    print "\n── Validation $SEP\n";
    if ($response->{valid} // 1) {
        print "Config is valid\n";
    } else {
        print "Warnings:\n";
        print "   * $_\n" for @{$response->{warnings} // []};
    }
}

sub print_generate_result {
    my ($response) = @_;
    if ($response->{headers_used} && @{$response->{headers_used}}) {
        print "Headers used : " . join(", ", @{$response->{headers_used}}) . "\n";
    }
    if ($response->{types_used} && %{$response->{types_used}}) {
        my %t = %{$response->{types_used}};
        print "Types used   : " . join(", ", map { "$_=$t{$_}" } sort keys %t) . "\n";
    }
    print "\n── Config $SEP\n";
    print $json->encode($response->{config});
    print "\n── Validation $SEP\n";
    if ($response->{valid}) {
        print "All column references are valid\n";
    } else {
        print "Column reference warnings:\n";
        print "   * $_\n" for @{$response->{warnings}};
        if ($response->{invalid_refs} && %{$response->{invalid_refs}}) {
            print "\n   Invalid refs: " . $json->encode($response->{invalid_refs}) . "\n";
        }
    }
}

sub print_modify_result {
    my ($original, $response, $instruction) = @_;
    my $changes = $response->{changes} // {};
    print "── Changes $SEP\n";
    print "   Instruction : $instruction\n";
    print "   Added       : " . ((@{$changes->{added}   // []}) ? join(", ", @{$changes->{added}})   : "none") . "\n";
    print "   Removed     : " . ((@{$changes->{removed} // []}) ? join(", ", @{$changes->{removed}}) : "none") . "\n";
    print "   Changed     : " . ((@{$changes->{changed} // []}) ? join(", ", @{$changes->{changed}}) : "none") . "\n";
    print "\n── Modified config $SEP\n";
    print $json->encode($response->{config});
    print "\n── Validation $SEP\n";
    if ($response->{valid}) {
        print "All column references are valid\n";
    } else {
        print "Column reference warnings:\n";
        print "   * $_\n" for @{$response->{warnings}};
    }
}

# ---------------------------------------------------------------------------
# MCP protocol
# ---------------------------------------------------------------------------

sub parse_extra_args {
    my @extra_args = @_;
    my ($headers, $data, $column_types);
    for my $arg (@extra_args) {
        if    ($arg =~ /^\{/) { $column_types = decode_json($arg) }
        elsif ($arg =~ /^\[/) { $data         = decode_json($arg) }
        else                  { $headers      = [split /,/, $arg] }
    }
    return ($headers, $data, $column_types);
}

sub call_tool {
    my ($tool_name, $args) = @_;

    # Initialize
    my $init = post_mcp(undef, {
        jsonrpc => "2.0", id => 1, method => "initialize",
        params  => {
            protocolVersion => "2025-11-25",
            capabilities    => {},
            clientInfo      => {name => "perl-client", version => "1.0.0"},
        },
    });
    die "Initialize failed: " . ($init->{error}{message} // "unknown") . "\n" if $init->{error};
    my $sid = $init->{_session_id};
    print "Connected  : $MCP_URL\n";

    # Notify
    post_mcp($sid, {jsonrpc => "2.0", method => "notifications/initialized", params => {}});

    # Call tool
    my $result = post_mcp($sid, {
        jsonrpc => "2.0", id => 2, method => "tools/call",
        params  => {name => $tool_name, arguments => $args},
    });
    die "Tool call failed: " . ($result->{error}{message} // "unknown") . "\n" if $result->{error};

    return decode_json($result->{result}{content}[0]{text});
}

sub post_mcp {
    my ($session_id, $payload) = @_;
    my $req = HTTP::Request->new(POST => $MCP_URL);
    $req->header("Content-Type"   => "application/json");
    $req->header("Accept"         => "application/json, text/event-stream");
    $req->header("Mcp-Session-Id" => $session_id) if $session_id;
    $req->content(JSON->new->utf8->encode($payload));
    my $res  = $ua->request($req);
    my $body = $res->decoded_content // $res->content // "";
    my $ct   = $res->content_type // "";
    my $new_sid = $res->header("Mcp-Session-Id");
    if ($ct =~ /event-stream/ || $body =~ /^data:/m) {
        my $js = _extract_sse_json($body) // return {};
        my $d  = eval { decode_json($js) }; warn "JSON error: $@\n" if $@; $d //= {};
        $d->{_session_id} = $new_sid if $new_sid;
        return $d;
    }
    return {_session_id => $new_sid} if $res->code == 202 && !$body;
    return {} unless $body && $body =~ /^\s*\{/;
    my $d = eval { decode_json($body) } // {};
    $d->{_session_id} = $new_sid if $new_sid;
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
