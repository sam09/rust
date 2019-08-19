//! Contains utilities for generating suggestions for borrowck errors related to unsatisified
//! outlives constraints.

use std::collections::BTreeMap;

use log::debug;
use rustc::{hir::def_id::DefId, infer::InferCtxt, mir::Body, ty::RegionVid};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Diagnostic, DiagnosticBuilder, Level};

use crate::borrow_check::nll::region_infer::{
    error_reporting::{
        region_name::{RegionName, RegionNameSource},
        ErrorConstraintInfo, ErrorReportingCtx, RegionErrorNamingCtx,
    },
    RegionInferenceContext,
};

/// Collects information about outlives constraints that needed to be added for a given MIR node
/// corresponding to a function definition.
///
/// Adds a help note suggesting adding a where clause with the needed constraints.
pub struct OutlivesSuggestionBuilder {
    /// The MIR DefId of the fn with the lifetime error.
    mir_def_id: DefId,

    /// The list of outlives constraints that need to be added. Specifically, we map each free
    /// region to all other regions that it must outlive. I will use the shorthand `fr:
    /// outlived_frs`. Not all of these regions will already have names necessarily. Some could be
    /// implicit free regions that we inferred. These will need to be given names in the final
    /// suggestion message.
    constraints_to_add: BTreeMap<RegionVid, Vec<RegionVid>>,
}

impl OutlivesSuggestionBuilder {
    /// Create a new builder for the given MIR node representing a fn definition.
    crate fn new(mir_def_id: DefId) -> Self {
        OutlivesSuggestionBuilder { mir_def_id, constraints_to_add: BTreeMap::default() }
    }

    /// Returns `true` iff the `RegionNameSource` is a valid source for an outlives
    /// suggestion.
    //
    // FIXME: Currently, we only report suggestions if the `RegionNameSource` is an early-bound
    // region or a named region, avoiding using regions with synthetic names altogether. This
    // allows us to avoid giving impossible suggestions (e.g. adding bounds to closure args).
    // We can probably be less conservative, since some inferred free regions are namable (e.g.
    // the user can explicitly name them. To do this, we would allow some regions whose names
    // come from `MatchedAdtAndSegment`, being careful to filter out bad suggestions, such as
    // naming the `'self` lifetime in methods, etc.
    fn region_name_is_suggestable(name: &RegionName) -> bool {
        match name.source {
            RegionNameSource::NamedEarlyBoundRegion(..)
            | RegionNameSource::NamedFreeRegion(..)
            | RegionNameSource::Static => {
                debug!("Region {:?} is suggestable", name);
                true
            }

            // Don't give suggestions for upvars, closure return types, or other unnamable
            // regions.
            RegionNameSource::SynthesizedFreeEnvRegion(..)
            | RegionNameSource::CannotMatchHirTy(..)
            | RegionNameSource::MatchedHirTy(..)
            | RegionNameSource::MatchedAdtAndSegment(..)
            | RegionNameSource::AnonRegionFromUpvar(..)
            | RegionNameSource::AnonRegionFromOutput(..)
            | RegionNameSource::AnonRegionFromYieldTy(..) => {
                debug!("Region {:?} is NOT suggestable", name);
                false
            }
        }
    }

    /// Returns a name for the region if it is suggestable. See `region_name_is_suggestable`.
    fn region_vid_to_name(
        &self,
        errctx: &ErrorReportingCtx<'_, '_, '_>,
        renctx: &mut RegionErrorNamingCtx,
        region: RegionVid,
    ) -> Option<RegionName> {
        errctx
            .rinfcx
            .give_region_a_name(errctx, renctx, region)
            .filter(Self::region_name_is_suggestable)
    }

    /// Add the outlives constraint `fr: outlived_fr` to the set of constraints we need to suggest.
    crate fn collect_constraint(&mut self, fr: RegionVid, outlived_fr: RegionVid) {
        debug!("Collected {:?}: {:?}", fr, outlived_fr);

        // Add to set of constraints for final help note.
        self.constraints_to_add.entry(fr).or_insert(Vec::new()).push(outlived_fr);
    }

    /// Emit an intermediate note on the given `Diagnostic` if the involved regions are
    /// suggestable.
    crate fn intermediate_suggestion(
        &mut self,
        errctx: &ErrorReportingCtx<'_, '_, '_>,
        errci: &ErrorConstraintInfo,
        renctx: &mut RegionErrorNamingCtx,
        diag: &mut DiagnosticBuilder<'_>,
    ) {
        // Emit an intermediate note.
        let fr_name = self.region_vid_to_name(errctx, renctx, errci.fr);
        let outlived_fr_name = self.region_vid_to_name(errctx, renctx, errci.outlived_fr);

        if let (Some(fr_name), Some(outlived_fr_name)) = (fr_name, outlived_fr_name) {
            if let RegionNameSource::Static = outlived_fr_name.source {
                diag.help(&format!("consider replacing `{}` with `'static`", fr_name));
            } else {
                diag.help(&format!(
                    "consider adding the following bound: `{}: {}`",
                    fr_name, outlived_fr_name
                ));
            }
        }
    }

    /// If there is a suggestion to emit, add a diagnostic to the buffer. This is the final
    /// suggestion including all collected constraints.
    crate fn add_suggestion<'tcx>(
        &self,
        body: &Body<'tcx>,
        rinfcx: &RegionInferenceContext<'tcx>,
        infcx: &InferCtxt<'_, 'tcx>,
        errors_buffer: &mut Vec<Diagnostic>,
        renctx: &mut RegionErrorNamingCtx,
    ) {
        // No constraints to add? Done.
        if self.constraints_to_add.is_empty() {
            debug!("No constraints to suggest.");
            return;
        }

        // If there is only one constraint to suggest, then we already suggested it in the
        // intermediate suggestion above.
        if self.constraints_to_add.len() == 1 {
            debug!("Only 1 suggestion. Skipping.");
            return;
        }

        // Create a new diagnostic.
        let mut diag = DiagnosticBuilder::new(
            infcx.tcx.sess.diagnostic(),
            Level::Help,
            "the following changes may resolve your lifetime errors",
        );

        let mir_span = infcx.tcx.def_span(self.mir_def_id);

        // Make sure that there is actually a suggestion to emit.
        let mut ndiags = 0;

        // We want this message to appear after other messages on the mir def.
        diag.sort_span = mir_span.shrink_to_hi();

        // Keep track of variables that we have already suggested unifying so that we don't print
        // out silly duplicate messages.
        let mut unified_already = FxHashSet::default();

        let errctx = ErrorReportingCtx {
            rinfcx,
            infcx,
            body,
            mir_def_id: self.mir_def_id,

            // We should not be suggesting naming upvars, so we pass in a dummy set of upvars that
            // should never be used.
            upvars: &[],
        };

        for (fr, outlived) in &self.constraints_to_add {
            let fr_name = if let Some(fr_name) = self.region_vid_to_name(&errctx, renctx, *fr) {
                fr_name
            } else {
                continue;
            };

            let outlived = outlived
                .iter()
                // if there is a `None`, we will just omit that constraint
                .filter_map(|fr| {
                    self.region_vid_to_name(&errctx, renctx, *fr).map(|rname| (fr, rname))
                })
                .collect::<Vec<_>>();

            // No suggestable outlived lifetimes.
            if outlived.is_empty() {
                continue;
            }

            // There are three types of suggestions we can make:
            // 1) Suggest a bound: 'a: 'b
            // 2) Suggest replacing 'a with 'static. If any of `outlived` is `'static`, then we
            //    should just replace 'a with 'static.
            // 3) Suggest unifying 'a with 'b if we have both 'a: 'b and 'b: 'a

            if outlived.iter().any(|(_, outlived_name)| {
                if let RegionNameSource::Static = outlived_name.source {
                    true
                } else {
                    false
                }
            }) {
                ndiags += 1;
                diag.help(&format!("replace `{}` with `'static`", fr_name));
            } else {
                // We want to isolate out all lifetimes that should be unified and print out
                // separate messages for them.

                let (unified, other): (Vec<_>, Vec<_>) = outlived.into_iter().partition(
                    // Do we have both 'fr: 'r and 'r: 'fr?
                    |(r, _)| {
                        self.constraints_to_add
                            .get(r)
                            .map(|r_outlived| r_outlived.as_slice().contains(fr))
                            .unwrap_or(false)
                    },
                );

                for (r, bound) in unified.into_iter() {
                    if !unified_already.contains(fr) {
                        ndiags += 1;
                        diag.help(&format!(
                            "`{}` and `{}` must be the same; replace one with the other",
                            fr_name, bound
                        ));
                        unified_already.insert(r);
                    }
                }

                if !other.is_empty() {
                    let other =
                        other.iter().map(|(_, rname)| format!("{}", rname)).collect::<Vec<_>>();
                    ndiags += 1;
                    diag.help(&format!("add bound `{}: {}`", fr_name, other.join(" + ")));
                }
            }
        }

        // If there is any suggestion, emit it. Otherwise, cancel.
        if ndiags > 0 {
            diag.buffer(errors_buffer);
        } else {
            diag.cancel();
        }
    }
}
