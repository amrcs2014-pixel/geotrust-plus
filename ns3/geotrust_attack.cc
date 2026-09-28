/*
 * GeoTrust+ end-to-end scenario, derived from the RangeGuard GRR attack-surface scenario (grr_attack.cc).
 * Extra modes: --mpr=grrw (GRR + overhearing watchdog + quarantine routing) and --mpr=geotrust (GeoTrust+:
 * PLR clamp, one-sided link test, watchdog, adaptive-f robust lazy-greedy MPR = MprMode 2, quarantine routing).
 *
 * OLSR with either the RFC 3626 MPR heuristic (--mpr=rfc) or the greedy round-robin (GRR) link-quality-constrained
 * MPR selection of Zhou et al., Computer Networks 289 (2026) 112648 (--mpr=grr, patched ns-3 OLSR, MprMode=1).
 * Channel: 802.11g ad-hoc MAC whose per-frame loss follows the paper's LoS UWB fit PLR(d) (as in rangeguard_ids.cc).
 *
 * GRR needs P_uw (u's own estimate, from its ranging to w) and P_wv (advertised by w in its ranging message; Zhou et
 * al. extend the ranging unit with a per-neighbour "Packet Loss Rate" field). Honest estimates are PLR(d_measured),
 * d_measured = true distance + N(0, sigmaR^2). The advertised field is unauthenticated, so a compromised node can lie.
 *
 * Attacks (start at tOn; attackers keep DEFAULT willingness unless stated):
 *   forge       advertise P_wv = forgeP (default 0) for every real link: relay capture through GRR, no fake links
 *   forge_adapt advertise the smallest value that passes every current neighbour's triangle bound (knows the defence)
 *   claim       classic reference: willingness ALWAYS + claim-all HELLOs (+ forged PLR)
 *   none
 *   All attackers drop transit data with --dropData and, as MPRs, do not retransmit TCs when --dropTc=1.
 *
 * Defence --defense=bound (deployable, attribution-free: the lie is in the advertiser's own message):
 *   u knows v is NOT its symmetric neighbour, hence d(u,v) >= dstar (calibrated), and u ranges to w, so by the
 *   triangle inequality d(w,v) >= dstar - d_meas(u,w) - margin =: dlb. An advertised P_wv < PLR(dlb) is impossible
 *   for an honest node (up to noise); u replaces it by PLR(dlb) and counts a violation against w.
 *
 * Summary (one CSV line per run): offered-load PDR after onset, mean MPR-set size, attacker share of MPR slots,
 * fraction of honest nodes that select an attacker, TC transmissions per node per second, bound violations
 * (honest / attacker advertisements). --calib=<csv> dumps d(u,v) of GRR two-hop pairs to calibrate dstar.
 */
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/olsr-module.h"
#include "ns3/propagation-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

using namespace ns3;

static double
PlrOfDistance(double d)
{
    return 1.0 / (std::exp(-1.61 * d + 9.97) + 1.01);
}

namespace ns3
{
class PlrLossModel : public PropagationLossModel
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::PlrLossModel")
                                .SetParent<PropagationLossModel>()
                                .SetGroupName("Propagation")
                                .AddConstructor<PlrLossModel>();
        return tid;
    }

    PlrLossModel()
    {
        m_rv = CreateObject<UniformRandomVariable>();
    }

  private:
    double DoCalcRxPower(double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override
    {
        return (m_rv->GetValue() < PlrOfDistance(a->GetDistanceFrom(b))) ? -300.0 : txPowerDbm - 30.0;
    }

    int64_t DoAssignStreams(int64_t stream) override
    {
        m_rv->SetStream(stream);
        return 1;
    }

    Ptr<UniformRandomVariable> m_rv;
};

NS_OBJECT_ENSURE_REGISTERED(PlrLossModel);
} // namespace ns3

namespace
{
constexpr double kInterval = 0.5; // HELLO/TC interval (s)

struct World
{
    uint32_t n{0}, tOn{60}, T{240}, warm{20};
    std::string attack{"none"}, defense{"none"};
    double forgeP{0.0}, sigmaR{0.08}, dstar{5.0}, margin{0.2};
    std::vector<Ptr<MobilityModel>> mob;
    std::vector<Ptr<olsr::RoutingProtocol>> rp;
    std::vector<char> attacker;
    std::mt19937_64 rng;
    std::normal_distribution<double> nd{0.0, 1.0};
    // metrics
    std::vector<double> sentK, recvK;
    double mprSum{0}, mprCnt{0}, attSlot{0}, allSlot{0}, selAtt{0}, selDen{0};
    double tcTx{0}, tcTxPre{0};
    double chkH{0}, violH{0}, chkA{0}, violA{0}, lieGap{0}, lieCnt{0};
    std::FILE* calib{nullptr};
    uint64_t calibRows{0};
    // ---- GeoTrust+ (defense == "geotrust" or "watchdog")
    double tauP{0.01}, linkRange{7.0}, wdZ{3.5};
    std::string fpolicy{"evidence"}; // adaptive-f trigger: evidence | suspicion | zero
    double tauR{0.05};                                          // reciprocity threshold [m]
    std::map<uint32_t, std::map<uint32_t, double>> fake;        // fixed fake links per insider -> claimed distance
    std::vector<int> partner;                                   // wormhole partner (-1 = none)
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> pending; // packet uid -> (observer, forwarder)
    std::map<std::pair<uint32_t, uint32_t>, std::array<double, 3>> wd; // (obs, fwd) -> k, mu, var
    std::set<std::pair<uint32_t, uint32_t>> flagged;               // watchdog verdicts
    double exclH{0}, exclHDen{0}, exclA{0}, exclADen{0}, fOn{0}, fDen{0};
    std::string topo;                                           // snapshot CSV (empty = off)
    uint32_t topoK{0};                                          // interval of the snapshot

    bool Active() const { return Simulator::Now().GetSeconds() >= tOn * kInterval; }
};

World* g = nullptr;

uint32_t
Idx(Ipv4Address a)
{
    return (a.Get() & 0xffff) - 1;
}

double
Dist(uint32_t a, uint32_t b)
{
    return g->mob[a]->GetDistanceFrom(g->mob[b]);
}

double
MeasD(uint32_t a, uint32_t b)
{
    return std::max(0.0, Dist(a, b) + g->sigmaR * g->nd(g->rng));
}

/** Smallest PLR an honest advertiser w can report for w -> v, as seen by u (triangle bound). */
double
Floor(uint32_t u, uint32_t w)
{
    double dlb = std::max(0.0, g->dstar - MeasD(u, w) - g->margin);
    return PlrOfDistance(dlb);
}

/** PLR value that w puts in its ranging message for its link to v. */
double
Advertised(uint32_t w, uint32_t v)
{
    double honest = PlrOfDistance(MeasD(w, v));
    if (!g->attacker[w] || !g->Active() || g->attack == "none")
    {
        return honest;
    }
    if (g->attack == "forge_adapt")
    {
        // broadcast value: must pass the bound of every current neighbour (the attacker ranges to all of them);
        // it never advertises a WORSE value than the truth
        double f = 0.0;
        for (uint32_t u = 0; u < g->n; u++)
        {
            if (u != w && u != v && Dist(u, w) < 8.0)
            {
                f = std::max(f, PlrOfDistance(std::max(0.0, g->dstar - Dist(u, w) - g->margin)));
            }
        }
        return std::min(honest, f);
    }
    return std::min(honest, g->forgeP); // forge, claim
}

double
GrrPlr(Ipv4Address obs, Ipv4Address from, Ipv4Address to)
{
    uint32_t u = Idx(obs), w = Idx(from), v = Idx(to);
    if (u >= g->n || w >= g->n || v >= g->n)
    {
        return 1.0;
    }
    if (u == w)
    {
        return PlrOfDistance(MeasD(u, v)); // own ranging-based estimate, not forgeable by others
    }
    if (g->calib && !g->attacker[u] && Simulator::Now().GetSeconds() > g->warm * kInterval)
    {
        std::fprintf(g->calib, "%.3f,%.3f,%.3f\n", Dist(u, v), Dist(u, w), Dist(w, v));
        g->calibRows++;
    }
    double adv = Advertised(w, v);
    bool fromAtt = g->attacker[w] && g->Active() && g->attack != "none";
    double honestTruth = PlrOfDistance(Dist(w, v));
    if (fromAtt)
    {
        g->lieGap += honestTruth - adv;
        g->lieCnt += 1;
    }
    if (g->defense == "bound" && !g->attacker[u])
    {
        double fl = Floor(u, w);
        bool viol = adv < fl - 1e-12;
        (fromAtt ? g->chkA : g->chkH) += 1;
        (fromAtt ? g->violA : g->violH) += viol ? 1 : 0;
        if (viol)
        {
            adv = fl;
        }
    }
    if (g->defense == "geotrust" && !g->attacker[u])
    {
        // GeoTrust+ PLR clamp: the ranging unit also carries w's distance to v, which reciprocity pins to
        // the truth (v measures the same exchange); a PLR better than that distance implies is clamped.
        (fromAtt ? g->chkA : g->chkH) += 1;
        double fl = PlrOfDistance(MeasD(w, v)) - g->tauP;
        if (adv < fl)
        {
            (fromAtt ? g->violA : g->violH) += 1;
            adv = fl;
        }
    }
    return adv;
}

// ------------------------------------------------------------------------------------------------------------
// Insider lies at the OLSR/ranging-unit level.  "True" adjacency for the lie/evidence model: Dist < linkRange.
//   claim      : every node advertised as symmetric neighbour (RangeGuard claim-all)
//   fakelinks  : A2 - 5 fixed real non-neighbours in (R, 2R] advertised, claimed at U(1, 0.6R)
//   stealth    : A12 - <= 3 fixed fakes in (R, 1.5R], claimed at U(0.75, 0.95) R
//   wormhole   : A9 - colluding pairs also advertise the partner's neighbours with the partner's distances
//   shrink     : A10b - distances to neighbours at >= R/2 claimed as 0.6 d; nodes in (R, 1.5R] advertised at 0.6 d
//   forge      : black-hole only for RFC 3626 (loss rates are ignored)
bool
IsLiar(uint32_t w)
{
    return g->attacker[w] && g->Active() &&
           (g->attack == "claim" || g->attack == "fakelinks" || g->attack == "stealth" ||
            g->attack == "wormhole" || g->attack == "shrink");
}

/** Fake neighbours advertised by insider w now (real nodes that are not within linkRange). */
std::vector<uint32_t>
FakeNbrs(uint32_t w)
{
    std::vector<uint32_t> out;
    if (!IsLiar(w))
    {
        return out;
    }
    const double R = g->linkRange;
    if (g->attack == "claim")
    {
        for (uint32_t v = 0; v < g->n; v++)
        {
            if (v != w && Dist(w, v) >= R)
            {
                out.push_back(v);
            }
        }
    }
    else if (g->attack == "fakelinks" || g->attack == "stealth")
    {
        const bool st = g->attack == "stealth";
        const double hi = st ? 1.5 * R : 2.0 * R;
        const size_t k = st ? 3 : 5;
        auto& fk = g->fake[w];
        for (auto it = fk.begin(); it != fk.end();)
        {
            double d = Dist(w, it->first);
            it = (d < R || d > hi) ? fk.erase(it) : std::next(it);
        }
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (uint32_t v = 0; v < g->n && fk.size() < k; v++)
        {
            double d = Dist(w, v);
            if (v != w && d >= R && d <= hi && !fk.count(v))
            {
                fk[v] = st ? (0.75 + 0.2 * U(g->rng)) * R : 1.0 + (0.6 * R - 1.0) * U(g->rng);
            }
        }
        for (const auto& kv : fk)
        {
            out.push_back(kv.first);
        }
    }
    else if (g->attack == "wormhole")
    {
        int p = g->partner[w];
        if (p >= 0)
        {
            for (uint32_t v = 0; v < g->n; v++)
            {
                if (v != w && int(v) != p && Dist(uint32_t(p), v) < R && Dist(w, v) >= R)
                {
                    out.push_back(v);
                }
            }
        }
    }
    else if (g->attack == "shrink")
    {
        for (uint32_t v = 0; v < g->n; v++)
        {
            double d = Dist(w, v);
            if (v != w && d >= R && d < 1.5 * R)
            {
                out.push_back(v);
            }
        }
    }
    return out;
}

std::vector<Ipv4Address>
HelloLies(Ipv4Address me)
{
    std::vector<Ipv4Address> out;
    uint32_t w = Idx(me);
    if (w >= g->n || g->attack == "claim")   // claim-all is emitted by the RangeGuard AttackClaimAll path
    {
        return out;
    }
    for (uint32_t v : FakeNbrs(w))
    {
        out.push_back(Ipv4Address((10u << 24) | (1u << 16) | (v + 1)));
    }
    return out;
}

/** Distance that w reports for w -> v in its ranging unit (honest: truth + 1 cm per-end noise). */
double
ClaimedDist(uint32_t w, uint32_t v)
{
    std::normal_distribution<double> E(0.0, 0.01);
    double d = Dist(w, v), noise = E(g->rng);
    if (!IsLiar(w))
    {
        return d + noise;
    }
    const double R = g->linkRange;
    if (g->attack == "shrink" && d >= 0.5 * R && d < 1.5 * R)
    {
        return 0.6 * d + noise;
    }
    if ((g->attack == "fakelinks" || g->attack == "stealth") && g->fake[w].count(v))
    {
        return g->fake[w][v] + noise;
    }
    if (g->attack == "wormhole" && g->partner[w] >= 0 && d >= R)
    {
        return Dist(uint32_t(g->partner[w]), v) + noise;
    }
    if (g->attack == "claim" && d >= R)
    {
        return 0.5 * R + noise;
    }
    return d + noise;
}

/** One-sided link claim visible to u: w advertises v, u hears v (v within R of u) and v does not list w. */
bool
OneSidedVisible(uint32_t u, uint32_t w)
{
    for (uint32_t v : FakeNbrs(w))
    {
        if (v != u && Dist(u, v) < g->linkRange)
        {
            return true;
        }
    }
    return false;
}

/** GeoTrust+ ranging evidence against neighbour w as seen by verifier u:
 *  own-link reciprocity, third-party reciprocity (degree >= 2 blame), one-sided claims (claimant blame).
 *  Only optimistic disagreements (claim shorter than the reference) count. */
bool
RangingConvicts(uint32_t u, uint32_t w)
{
    const double R = g->linkRange, tau = g->tauR;
    if (OneSidedVisible(u, w))
    {
        return true;
    }
    if (Dist(u, w) < R && ClaimedDist(u, w) - ClaimedDist(w, u) > tau)
    {
        return true;                                    // w claims to be closer to u than u measures
    }
    int bad = 0;
    for (uint32_t v = 0; v < g->n && bad < 2; v++)
    {
        if (v == u || v == w || Dist(u, v) >= R || Dist(w, v) >= R)
        {
            continue;                                   // common neighbours only (u hears v's report)
        }
        if (ClaimedDist(v, w) - ClaimedDist(w, v) > tau)
        {
            bad++;
        }
    }
    return bad >= 2;
}

bool
Excluded(Ipv4Address me, Ipv4Address other)
{
    uint32_t u = Idx(me), w = Idx(other);
    if (u >= g->n || w >= g->n || g->attacker[u])
    {
        return false;
    }
    const std::string& d = g->defense;
    if (d == "topo")
    {
        return OneSidedVisible(u, w);                   // topological reciprocity only
    }
    bool wdFlag = g->flagged.count({u, w}) > 0;
    if (d == "geotrust")
    {
        return wdFlag || RangingConvicts(u, w);
    }
    return wdFlag;                                      // watchdog
}

uint32_t
RobustF(Ipv4Address me)
{
    // adaptive robustness: f = 1 only where this node has evidence of misbehaviour nearby
    uint32_t u = Idx(me);
    bool ev = false;
    for (uint32_t w = 0; w < g->n && !ev && g->fpolicy != "zero"; w++)
    {
        if (w == u || Dist(u, w) >= g->linkRange)
        {
            continue;
        }
        if (g->fpolicy == "suspicion")
        {
            // unexplained risk only: a relay whose overhearing statistic is sliding but not yet convicted
            auto it = g->wd.find({u, w});
            if (it != g->wd.end() && !g->flagged.count({u, w}) && it->second[1] >= 3)
            {
                const auto& s = it->second;
                ev = (s[0] - s[1]) / std::sqrt(s[2] + 1.0) < -1.5;
            }
        }
        else if (Excluded(me, Ipv4Address((10u << 24) | (1u << 16) | (w + 1))))
        {
            ev = true; // "evidence": any exclusion in the neighbourhood
        }
    }
    if (g->Active())
    {
        g->fOn += ev ? 1 : 0;
        g->fDen += 1;
    }
    return ev ? 1 : 0;
}

void
WdSend(Ipv4Address me, Ipv4Address next, Ipv4Address, uint32_t uid)
{
    uint32_t u = Idx(me), w = Idx(next);
    if (u < g->n && w < g->n && !g->attacker[u])
    {
        g->pending[uid] = {u, w};
    }
}

void
WdForward(Ipv4Address me, uint32_t uid, bool forwarded)
{
    auto it = g->pending.find(uid);
    uint32_t w = Idx(me);
    if (it == g->pending.end() || it->second.second != w)
    {
        return;
    }
    uint32_t u = it->second.first;
    g->pending.erase(it);
    double p = 1.0 - PlrOfDistance(Dist(u, w)); // chance that u overhears w's retransmission
    std::uniform_real_distribution<double> U(0.0, 1.0);
    bool heard = forwarded && U(g->rng) < p;
    auto& s = g->wd[{u, w}];
    s[0] += heard ? 1 : 0;
    s[1] += p;
    s[2] += p * (1 - p);
    if (s[1] >= 10 && (s[0] - s[1]) / std::sqrt(s[2] + 1.0) < -g->wdZ)
    {
        g->flagged.insert({u, w});
    }
}

double
HopCost(Ipv4Address, Ipv4Address)
{
    return 1.0; // hop-count routes; the patch's Dijkstra then honours the quarantine predicate
}

void
OnDataTx(Ptr<const Packet>)
{
    size_t k = size_t(Simulator::Now().GetSeconds() / kInterval);
    if (k < g->sentK.size())
    {
        g->sentK[k] += 1;
    }
}

void
OnDataRx(Ptr<const Packet>, const Address&)
{
    size_t k = size_t(Simulator::Now().GetSeconds() / kInterval);
    if (k < g->recvK.size())
    {
        g->recvK[k] += 1;
    }
}

void
OnOlsrTx(const olsr::PacketHeader&, const olsr::MessageList& msgs)
{
    for (const auto& m : msgs)
    {
        if (m.GetMessageType() == olsr::MessageHeader::TC_MESSAGE)
        {
            (g->Active() ? g->tcTx : g->tcTxPre) += 1;
        }
    }
}

/** Topology snapshot: node positions and roles, MPR sets, advertised fake links, exclusions. */
void
DumpTopology(uint32_t k)
{
    std::FILE* f = std::fopen(g->topo.c_str(), "w");
    std::fprintf(f, "kind,a,b,x,y,z\n");
    for (uint32_t i = 0; i < g->n; i++)
    {
        Vector p = g->mob[i]->GetPosition();
        std::fprintf(f, "node,%u,%d,%.3f,%.3f,%.3f\n", i, int(g->attacker[i]), p.x, p.y, p.z);
    }
    for (uint32_t i = 0; i < g->n; i++)
    {
        for (const auto& a : g->rp[i]->GetMprSet())
        {
            std::fprintf(f, "mpr,%u,%u,,,\n", i, Idx(a));
        }
        for (uint32_t v : FakeNbrs(i))
        {
            std::fprintf(f, "fake,%u,%u,,,\n", i, v);
        }
        if (g->attacker[i])
        {
            continue;
        }
        Ipv4Address me((10u << 24) | (1u << 16) | (i + 1));
        for (uint32_t j = 0; j < g->n; j++)
        {
            if (j != i && Dist(i, j) < g->linkRange &&
                Excluded(me, Ipv4Address((10u << 24) | (1u << 16) | (j + 1))))
            {
                std::fprintf(f, "excl,%u,%u,,,\n", i, j);
            }
        }
    }
    std::fprintf(f, "meta,%u,%u,%.3f,,\n", k, g->n, g->linkRange);
    std::fclose(f);
}

void
Tick(uint32_t k)
{
    if (!g->topo.empty() && k == g->topoK)
    {
        DumpTopology(k);
    }
    if (k >= g->warm)
    {
        bool post = k >= g->tOn + 20;
        for (uint32_t i = 0; i < g->n; i++)
        {
            if (g->attacker[i])
            {
                continue;
            }
            auto S = g->rp[i]->GetMprSet();
            if (post || g->attack == "none")
            {
                g->mprSum += S.size();
                g->mprCnt += 1;
            }
            if (!post)
            {
                continue;
            }
            bool hasAtt = false, nbAtt = false;
            for (const auto& a : S)
            {
                uint32_t j = Idx(a);
                g->allSlot += 1;
                if (j < g->n && g->attacker[j])
                {
                    g->attSlot += 1;
                    hasAtt = true;
                }
            }
            for (uint32_t j = 0; j < g->n; j++)
            {
                if (g->attacker[j] && Dist(i, j) < 5.0)
                {
                    nbAtt = true;
                }
            }
            if (nbAtt)
            {
                g->selDen += 1;
                g->selAtt += hasAtt ? 1 : 0;
            }
            // GeoTrust+ exclusion bookkeeping (honest verifier i, neighbour j)
            Ipv4Address me((10u << 24) | (1u << 16) | (i + 1));
            for (uint32_t j = 0; j < g->n; j++)
            {
                if (j == i || Dist(i, j) >= g->linkRange)
                {
                    continue;
                }
                bool ex = Excluded(me, Ipv4Address((10u << 24) | (1u << 16) | (j + 1)));
                if (g->attacker[j])
                {
                    g->exclA += ex;
                    g->exclADen += 1;
                }
                else
                {
                    g->exclH += ex;
                    g->exclHDen += 1;
                }
            }
        }
    }
    if (k + 1 < g->T)
    {
        Simulator::Schedule(Seconds(kInterval), &Tick, k + 1);
    }
}
} // namespace

int
main(int argc, char* argv[])
{
    World w;
    g = &w;
    uint32_t n = 30, retries = 1, mbps = 6, seed = 1, run = 1, dropTc = 1, qmin = 1;
    double vmax = 0.6, L = 15.0, H = 3.0, frac = 0.1, gamma = 1.2, dropData = 1.0, absT = 0.0;
    std::string mpr = "grr", calib, summary = "geotrust_summary.csv";

    CommandLine cmd(__FILE__);
    cmd.AddValue("n", "number of drones", n);
    cmd.AddValue("L", "arena side (m)", L);
    cmd.AddValue("H", "arena height (m)", H);
    cmd.AddValue("vmax", "max speed (m/s)", vmax);
    cmd.AddValue("mpr", "rfc|grr", mpr);
    cmd.AddValue("gamma", "GRR gain factor", gamma);
    cmd.AddValue("qmin", "0 = Q_min over all relays, 1 = best single relay", qmin);
    cmd.AddValue("fpolicy", "GeoTrust+ adaptive-f trigger: evidence|suspicion|zero", w.fpolicy);
    cmd.AddValue("absT", "GRR absolute two-hop PLR target (0 = paper)", absT);
    cmd.AddValue("attack", "none|forge|forge_adapt|claim|fakelinks|stealth|wormhole|shrink", w.attack);
    cmd.AddValue("frac", "attacker fraction", frac);
    cmd.AddValue("forgeP", "PLR advertised by a forging attacker", w.forgeP);
    cmd.AddValue("dropData", "attacker transit-data drop probability", dropData);
    cmd.AddValue("dropTc", "1 = attacker MPR does not retransmit TCs", dropTc);
    cmd.AddValue("defense", "none|bound", w.defense);
    cmd.AddValue("dstar", "minimum distance of a non-neighbour (m), calibrated", w.dstar);
    cmd.AddValue("margin", "ranging margin of the triangle bound (m)", w.margin);
    cmd.AddValue("sigmaR", "ranging noise std (m)", w.sigmaR);
    cmd.AddValue("retries", "802.11 retransmissions", retries);
    cmd.AddValue("mbps", "802.11g OFDM rate", mbps);
    cmd.AddValue("T", "intervals to simulate", w.T);
    cmd.AddValue("tOn", "attack onset interval", w.tOn);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("run", "RNG run", run);
    cmd.AddValue("calib", "dump two-hop distances to this CSV", calib);
    cmd.AddValue("summary", "summary CSV (appended)", summary);
    cmd.AddValue("topo", "topology snapshot CSV (empty = off)", w.topo);
    cmd.AddValue("topoK", "interval of the topology snapshot", w.topoK);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);
    Config::SetDefault("ns3::WifiMac::FrameRetryLimit", UintegerValue(retries + 1));
    w.rng.seed(uint64_t(seed) * 7919 + run);
    w.n = n;
    w.sentK.assign(w.T + 2, 0.0);
    w.recvK.assign(w.T + 2, 0.0);
    if (!calib.empty())
    {
        w.calib = std::fopen(calib.c_str(), "w");
        std::fprintf(w.calib, "d_uv,d_uw,d_wv\n");
    }

    // ---- roles
    uint32_t na = (w.attack == "none") ? 0 : std::max(1u, uint32_t(std::lround(frac * n)));
    std::vector<uint32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), w.rng);
    w.attacker.assign(n, 0);
    std::vector<uint32_t> att(perm.begin(), perm.begin() + std::min(na, n));
    for (uint32_t a : att)
    {
        w.attacker[a] = 1;
    }
    w.partner.assign(n, -1);                            // wormhole colluders are paired in draw order
    for (size_t q = 0; q + 1 < att.size(); q += 2)
    {
        w.partner[att[q]] = int(att[q + 1]);
        w.partner[att[q + 1]] = int(att[q]);
    }

    // ---- network
    NodeContainer nodes;
    nodes.Create(n);
    ObjectFactory posFactory;
    posFactory.SetTypeId("ns3::RandomBoxPositionAllocator");
    posFactory.Set("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(L) + "]"));
    posFactory.Set("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(L) + "]"));
    posFactory.Set("Z", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(H) + "]"));
    Ptr<PositionAllocator> alloc = posFactory.Create<PositionAllocator>();
    MobilityHelper mobility;
    mobility.SetPositionAllocator(alloc);
    std::ostringstream speed;
    speed << "ns3::UniformRandomVariable[Min=0.2|Max=" << std::max(vmax, 0.2001) << "]";
    mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                              "Speed", StringValue(speed.str()),
                              "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"),
                              "PositionAllocator", PointerValue(alloc));
    mobility.Install(nodes);

    Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel>();
    channel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());
    channel->SetPropagationLossModel(CreateObject<PlrLossModel>());
    YansWifiPhyHelper phy;
    phy.SetChannel(channel);
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("ErpOfdmRate" + std::to_string(mbps) + "Mbps"),
                                 "ControlMode", StringValue("ErpOfdmRate" + std::to_string(mbps) + "Mbps"));
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    InternetStackHelper internet;
    OlsrHelper olsr;
    olsr.Set("HelloInterval", TimeValue(Seconds(kInterval)));
    olsr.Set("TcInterval", TimeValue(Seconds(kInterval)));
    olsr.Set("MidInterval", TimeValue(Seconds(kInterval * 10)));
    olsr.Set("HnaInterval", TimeValue(Seconds(kInterval * 10)));
    // modes: rfc | rfct (RFC + topology check) | rfcw (RFC + watchdog) | rfcg (RFC + GeoTrust+)
    //        grr | grrw | geotrust (GRR-based).  All defended modes share quarantine-aware route computation.
    const bool defended = (mpr == "geotrust" || mpr == "grrw" || mpr == "rfcw" || mpr == "rfcg" || mpr == "rfct");
    if (mpr == "geotrust" || mpr == "rfcg")
    {
        w.defense = "geotrust";
    }
    else if (mpr == "grrw" || mpr == "rfcw")
    {
        w.defense = "watchdog";
    }
    else if (mpr == "rfct")
    {
        w.defense = "topo";
    }
    uint32_t mode = 1;
    if (mpr == "rfc" || mpr == "rfcw")
    {
        mode = 0;
    }
    else if (mpr == "geotrust")
    {
        mode = 2;
    }
    else if (mpr == "rfcg" || mpr == "rfct")
    {
        mode = 3;
    }
    olsr.Set("MprMode", UintegerValue(mode));
    olsr.Set("LinkQualityAware", BooleanValue(defended));
    olsr.Set("GrrGamma", DoubleValue(gamma));
    olsr.Set("GrrQmin", UintegerValue(qmin));
    olsr.Set("GrrAbsTarget", DoubleValue(absT));
    internet.SetRoutingHelper(olsr);
    internet.Install(nodes);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.0.0", "255.255.0.0");
    Ipv4InterfaceContainer ifaces = ipv4.Assign(devices);
    for (uint32_t i = 0; i < n; i++)
    {
        w.mob.push_back(nodes.Get(i)->GetObject<MobilityModel>());
        w.rp.push_back(Ipv4RoutingHelper::GetRouting<olsr::RoutingProtocol>(
            nodes.Get(i)->GetObject<Ipv4>()->GetRoutingProtocol()));
        w.rp.back()->TraceConnectWithoutContext("Tx", MakeCallback(&OnOlsrTx));
    }
    olsr::RoutingProtocol::SetGrrPlrFunction(&GrrPlr);
    olsr::RoutingProtocol::SetHelloLiesFunction(&HelloLies);
    if (defended)
    {
        olsr::RoutingProtocol::SetGeoTrustFunctions(&Excluded, &RobustF);
        olsr::RoutingProtocol::SetQuarantineFunction(&Excluded);
        olsr::RoutingProtocol::SetLinkCostFunction(&HopCost);
        olsr::RoutingProtocol::SetDataHooks(&WdSend, &WdForward);
    }

    // ---- attack activation
    if (!att.empty())
    {
        std::vector<Ipv4Address> all;
        for (uint32_t i = 0; i < n; i++)
        {
            all.push_back(ifaces.GetAddress(i));
        }
        olsr::RoutingProtocol::SetAllAddresses(all);
        bool claim = w.attack == "claim";
        for (uint32_t a : att)
        {
            Ptr<olsr::RoutingProtocol> r = w.rp[a];
            Simulator::Schedule(Seconds(w.tOn * kInterval), [r, dropData, dropTc, claim]() {
                r->SetAttribute("AttackDropProb", DoubleValue(dropData));
                r->SetAttribute("AttackDropControl", BooleanValue(dropTc != 0));
                if (claim)
                {
                    r->SetAttribute("Willingness", StringValue("always"));
                    r->SetAttribute("AttackClaimAll", BooleanValue(true));
                }
            });
        }
    }

    // ---- data traffic between honest nodes (offered-load PDR)
    std::vector<uint32_t> benign;
    for (uint32_t i = 0; i < n; i++)
    {
        if (!w.attacker[i])
        {
            benign.push_back(i);
        }
    }
    Ptr<UniformRandomVariable> pick = CreateObject<UniformRandomVariable>();
    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), 9000));
    sink.Install(nodes).Start(Seconds(0));
    uint32_t flows = 0;
    for (uint32_t i : benign)
    {
        uint32_t j = benign[pick->GetInteger(0, benign.size() - 1)];
        if (j == i)
        {
            j = benign[(std::find(benign.begin(), benign.end(), j) - benign.begin() + 1) % benign.size()];
        }
        OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(ifaces.GetAddress(j), 9000));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetConstantRate(DataRate("2048bps"), 64);
        ApplicationContainer a = onoff.Install(nodes.Get(i));
        a.Start(Seconds(3 + pick->GetValue(0, 1)));
        a.Stop(Seconds(w.T * kInterval - 1));
        flows++;
    }
    Config::ConnectWithoutContext("/NodeList/*/ApplicationList/*/$ns3::OnOffApplication/Tx", MakeCallback(&OnDataTx));
    Config::ConnectWithoutContext("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback(&OnDataRx));

    Simulator::Schedule(Seconds(kInterval), &Tick, 1u);
    Simulator::Stop(Seconds(w.T * kInterval + 0.01));
    Simulator::Run();
    Simulator::Destroy();
    if (w.calib)
    {
        std::fclose(w.calib);
    }

    // offered-load PDR: every flow offers 2 packets per interval; window [tOn + 20, T - 2)
    auto offered = [&](uint32_t lo, uint32_t hi) {
        double r = 0;
        for (uint32_t q = lo; q < hi; q++)
        {
            r += w.recvK[q];
        }
        return r / (2.0 * flows * (hi - lo));
    };
    double pdrPre = offered(20, w.tOn), pdrPost = offered(w.tOn + 20, w.T - 2);
    double secPost = (w.T - w.tOn) * kInterval, secPre = w.tOn * kInterval;

    bool newFile = !std::ifstream(summary).good();
    std::ofstream f(summary, std::ios::app);
    if (newFile)
    {
        f << "run,seed,n,L,vmax,mpr,gamma,absT,attack,frac,forgeP,dropData,dropTc,defense,dstar,margin,sigmaR,attackers,"
             "pdr_pre,pdr_post,mpr_size,att_slot_share,sel_att_frac,tc_per_node_s_pre,tc_per_node_s_post,"
             "viol_honest,viol_attacker,checks_honest,checks_attacker,mean_lie_gap,excl_honest,excl_attacker,f_on\n";
    }
    f << run << ',' << seed << ',' << n << ',' << L << ',' << vmax << ',' << mpr << ',' << gamma << ',' << absT << ',' << w.attack << ','
      << frac << ',' << w.forgeP << ',' << dropData << ',' << dropTc << ',' << w.defense << ',' << w.dstar << ','
      << w.margin << ',' << w.sigmaR << ',' << na << ',' << pdrPre << ',' << pdrPost << ','
      << (w.mprCnt > 0 ? w.mprSum / w.mprCnt : 0.0) << ',' << (w.allSlot > 0 ? w.attSlot / w.allSlot : 0.0) << ','
      << (w.selDen > 0 ? w.selAtt / w.selDen : 0.0) << ',' << w.tcTxPre / (n * secPre) << ','
      << w.tcTx / (n * secPost) << ',' << (w.chkH > 0 ? w.violH / w.chkH : 0.0) << ','
      << (w.chkA > 0 ? w.violA / w.chkA : 0.0) << ',' << w.chkH << ',' << w.chkA << ','
      << (w.lieCnt > 0 ? w.lieGap / w.lieCnt : 0.0) << ',' << (w.exclHDen > 0 ? w.exclH / w.exclHDen : 0.0) << ','
      << (w.exclADen > 0 ? w.exclA / w.exclADen : 0.0) << ',' << (w.fDen > 0 ? w.fOn / w.fDen : 0.0) << '\n';
    std::cout << mpr << ' ' << w.attack << " def=" << w.defense << " n=" << n << " L=" << L << " pdr_pre=" << pdrPre
              << " pdr_post=" << pdrPost << " mpr=" << (w.mprCnt > 0 ? w.mprSum / w.mprCnt : 0.0)
              << " attShare=" << (w.allSlot > 0 ? w.attSlot / w.allSlot : 0.0) << " calibRows=" << w.calibRows
              << std::endl;
    return 0;
}
