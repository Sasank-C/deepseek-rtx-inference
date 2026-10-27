#!/usr/bin/env python3
"""Read-only container networking inspection; never creates interfaces or keys.

Two-node inspection needs an operator-reviewed proof JSON with mode, machine_ids,
and cluster_ids (overlay) or vm_eligible (wireguard). PASS_ROUTE_ONLY is NOT
permission to download weights: the real eight-rank collective gate remains.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess


def proof_valid(proof):
    machines = proof.get('machine_ids', [])
    if len(machines) != 2 or any(type(x) is not int or x <= 0 for x in machines) or machines[0] == machines[1]:
        return False
    if proof.get('mode') == 'overlay':
        clusters = proof.get('cluster_ids', [])
        return (len(clusters) == 2 and clusters[0] is not None
                and str(clusters[0]) not in ('', 'None', 'null')
                and clusters[0] == clusters[1])
    return proof.get('mode') == 'wireguard' and proof.get('vm_eligible') == [True, True]


def command(args):
    return subprocess.check_output(args, timeout=5, text=True, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proof', type=Path)
    p.add_argument('--private-iface')
    p.add_argument('--peer-ip')
    args = p.parse_args()
    status = Path('/proc/self/status').read_text()
    cap = int(re.search(r'^CapEff:\s*([0-9a-fA-F]+)$', status, re.M)[1], 16)
    report = {'uid': os.getuid(), 'cap_net_admin': bool(cap & (1 << 12)),
              'tun_device_exists': Path('/dev/net/tun').exists(),
              'gate': 'DENY_TWO_NODE', 'collectives_tested': False,
              'note': 'UID/TUN presence alone does not establish a usable private route'}
    try:
        if not all((args.proof, args.private_iface, args.peer_ip)):
            raise ValueError('Need supported-offer proof, private interface, and peer IP')
        proof = json.loads(args.proof.read_text())
        if not proof_valid(proof):
            raise ValueError('No same-cluster overlay or two-VM eligibility proof')
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+', args.private_iface) or args.private_iface == 'lo':
            raise ValueError('Invalid private interface')
        peer = ipaddress.IPv4Address(args.peer_ip)
        if not peer.is_private or peer.is_loopback or peer.is_unspecified or peer.is_link_local:
            raise ValueError('Peer must be a private unicast IPv4 address')
        route = json.loads(command(['ip', '-j', 'route', 'get', str(peer)]))[0]
        if route.get('dev') != args.private_iface or route.get('gateway'):
            raise ValueError('Peer is not directly routed on the specified private interface')
        addresses = json.loads(command(['ip', '-j', 'address', 'show', 'dev', args.private_iface]))
        own = [v['local'] for row in addresses for v in row.get('addr_info', [])
               if v.get('family') == 'inet']
        if not own or str(peer) in own:
            raise ValueError('Missing own address or peer resolves to this node')
        if proof['mode'] == 'wireguard':
            if not report['cap_net_admin']:
                raise ValueError('Kernel VPN requires network administration capability')
            links = json.loads(command(['ip', '-d', '-j', 'link', 'show', 'dev', args.private_iface]))
            if links[0].get('linkinfo', {}).get('info_kind') != 'wireguard':
                raise ValueError('Expected an actual kernel WireGuard interface')
            # Report only a boolean; never run showconf or dump private keys.
            import time
            handshakes = command(['wg', 'show', args.private_iface, 'latest-handshakes'])
            stamps = [int(line.split()[-1]) for line in handshakes.splitlines() if line.strip()]
            if len(stamps) != 1 or not 0 <= time.time() - stamps[0] < 180:
                raise ValueError('Expected one authenticated, recent WireGuard peer')
            report['recent_peer_handshake'] = True
        report.update(gate='PASS_ROUTE_ONLY', mode=proof['mode'],
                      private_interface=args.private_iface, own_ipv4=own, peer_ipv4=str(peer))
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as exc:
        report['reason'] = str(exc)[:240]
    print(json.dumps(report))
    raise SystemExit(0 if report['gate'] == 'PASS_ROUTE_ONLY' else 2)


if __name__ == '__main__':
    main()
