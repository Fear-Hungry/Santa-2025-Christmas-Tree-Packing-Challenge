#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "geom.hpp"

class SARefiner {
public:
    enum class OverlapMetric {
        kArea = 0,
        kMtv2 = 1,
    };

    struct Params {
        int iters = 0;
        double t0 = 0.15;
        double t1 = 0.01;
        double p_rot = 0.30;
        double ddeg_max = 20.0;
        double ddeg_min = 2.0;
        double step_frac_max = 0.10;
        double step_frac_min = 0.004;
        double p_pick_extreme = 0.95;
        int extreme_topk = 14;
        int rebuild_extreme_every = 25;
        double p_random_dir = 0.15;
        double kick_prob = 0.02;
        double kick_mult = 3.0;
        // Reheat on stagnation: after `reheat_iters` without best improvement.
        int reheat_iters = 0;
        double reheat_mult = 1.0;
        double reheat_step_mult = 1.0;
        int reheat_max = 1;

        // Quantização no espaço do CSV. Se `quantize_decimals >= 0`, o SA
        // avalia candidatos já quantizados e mantém o estado quantizado após
        // movimentos aceitos, evitando "ganhos" que somem no arredondamento do
        // submission. Use -1 para desligar.
        int quantize_decimals = 9;

        // Portfólio de movimentos ("hiperheurística"): pesos iniciais.
        // Setar um peso como 0 desliga o operador.
        double w_micro = 1.0;
        double w_swap_rot = 0.25;
        double w_relocate = 0.15;
        double w_block_translate = 0.05;
        double w_block_rotate = 0.02;
        double w_lns = 0.001;
        double w_push_contact = 0.0;
        double w_squeeze = 0.0;

        // Controlador adaptativo (ALNS-style).
        int hh_segment = 50;
        double hh_reaction = 0.20;
	        double hh_min_weight = 0.05;
	        double hh_max_block_weight = 0.15;
	        double hh_max_lns_weight = 0.05;
	        // Escala do reward: recompensa proporcional a Δs / custo do operador.
	        double hh_reward_scale = 1000.0;
	        double hh_reward_best = 5.0;
        double hh_reward_improve = 2.0;
        double hh_reward_accept = 0.0;

        // Relocate (macro-movimento).
        int relocate_attempts = 10;
        double relocate_pull_min = 0.50;
        double relocate_pull_max = 1.00;
        double relocate_noise_frac = 0.08;  // relativo a curr_side
        double relocate_p_rot = 0.70;

        // Block moves (macro): move um subconjunto coerente (vizinhos próximos).
        int block_size = 6;
        double block_step_frac_max = 0.25;
        double block_step_frac_min = 0.03;
        double block_p_random_dir = 0.10;
        double block_rot_deg_max = 25.0;
        double block_rot_deg_min = 3.0;

        // LNS (macro): remove um subconjunto da borda e reinsere por amostragem.
        int lns_remove = 6;
        int lns_attempts_per_tree = 30;
        double lns_p_uniform = 0.30;
        double lns_p_contact = 0.35;
        double lns_pull_min = 0.30;
        double lns_pull_max = 0.90;
        double lns_noise_frac = 0.15;  // relativo a curr_side
        double lns_p_rot = 0.60;
        double lns_box_mult = 1.05;

        // Soft constraints (overlap): quando overlap_weight > 0, o SA pode aceitar
        // overlaps temporários, penalizando o overlap conforme a métrica escolhida.
        // O retorno do SA continua sendo a melhor solução válida (overlap ~ 0).
        OverlapMetric overlap_metric = OverlapMetric::kArea;
        double overlap_weight = 0.0;      // 0 => hard constraint (padrão)
        double overlap_weight_start = -1.0;  // < 0 => usa overlap_weight
        double overlap_weight_end = -1.0;    // < 0 => usa overlap_weight
        double overlap_weight_power = 1.0;   // 1 => linear
        double overlap_eps_area = 1e-12;     // métrica <= eps conta como "sem overlap"
        double overlap_cost_cap = 0.0;    // 0 => sem cap; senão, rejeita custo acima do cap

        // Tie-breaker em platôs de `side = max(width, height)`: adiciona um termo
        // secundário `plateau_eps * min(width, height)` ao custo para dar sinal
        // quando só o eixo "não-dominante" muda.
        double plateau_eps = 0.0;

        // Operador extra para resolver overlaps (repulsão do centroide da interseção).
        double w_resolve_overlap = 0.0;
        int resolve_attempts = 6;
        double resolve_step_frac_max = 0.20;
        double resolve_step_frac_min = 0.02;
        double resolve_noise_frac = 0.05;

        // Push-to-contact (line-search) no eixo dominante (hard constraint).
        // Move uma árvore da casca ao longo de ±x/±y até o primeiro contato.
        double push_max_step_frac = 0.60;  // limite do step inicial (relativo a curr_side)
        int push_bisect_iters = 10;        // número de checagens na busca binária
        double push_overshoot_frac = 0.0;  // fração de max_step para "atravessar" o contato (soft)

        // Squeeze: repete `push_to_contact` algumas vezes no eixo dominante.
        int squeeze_pushes = 6;
    };

    struct Result {
        std::vector<TreePose> best_poses;
        double best_side = std::numeric_limits<double>::infinity();
    };

    SARefiner(const Polygon& base_poly, double radius);

    Result refine_min_side(const std::vector<TreePose>& start,
                           uint64_t seed,
                           const Params& p,
                           const std::vector<char>* active_mask = nullptr) const;

    static void apply_aggressive_preset(Params& p);

    // MTV (minimum translation vector) aproximado via decomposição em triângulos + SAT.
    // Retorna um vetor para mover `a` para fora de `b` (0/false se não houver overlap).
    bool overlap_mtv(const TreePose& a,
                     const TreePose& b,
                     Point& out_mtv,
                     double& out_overlap_area) const;

private:
    const Polygon& base_poly_;
    double radius_;
    std::vector<std::array<Point, 3>> base_tris_;

    struct OverlapInfo {
        double area = 0.0;
        Point centroid{0.0, 0.0};
    };

    struct OverlapSeparation {
        double area = 0.0;
        Point mtv_sum{0.0, 0.0};  // soma (MTV * área) para mover `a` para fora de `b`
    };

    OverlapInfo overlap_info(const TreePose& a, const TreePose& b) const;
    OverlapSeparation overlap_separation(const TreePose& a, const TreePose& b) const;
};
