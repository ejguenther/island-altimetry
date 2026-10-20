"""Outcome codes. An API error must never become a scientific zero."""

OK = "ok"
NO_GRANULES = "no_granules"
NO_SAMPLES_IN_AOI = "no_samples_in_aoi"
ALL_FILTERED = "all_filtered"
SERVICE_UNSUPPORTED = "service_unsupported"
FIELD_UNSUPPORTED = "field_unsupported"
RETRIEVAL_FAILED = "retrieval_failed"
NOT_YET_QUERIED = "not_yet_queried"

ALL = (OK, NO_GRANULES, NO_SAMPLES_IN_AOI, ALL_FILTERED, SERVICE_UNSUPPORTED,
       FIELD_UNSUPPORTED, RETRIEVAL_FAILED, NOT_YET_QUERIED)

# Outcomes that are scientific findings (complete); everything else is unresolved.
TERMINAL_EMPTY = (NO_GRANULES, NO_SAMPLES_IN_AOI, ALL_FILTERED)
