package com.bittupandit.ssi;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ModuleUrlValidatorTest {
    @Test
    public void acceptsOnlyProjectModuleDocsOnApprovedBranches() {
        assertTrue(ModuleUrlValidator.isAllowed(
            "https://github.com/egalego-sudo/special-super-intelligency/blob/android-control-app-v1/docs/modules/01-independent-foundation-model.md"
        ));
        assertTrue(ModuleUrlValidator.isAllowed(
            "https://github.com/egalego-sudo/special-super-intelligency/blob/main/docs/modules/14-scientific-proof.md"
        ));
    }

    @Test
    public void rejectsExternalHostsAndUnapprovedSchemes() {
        assertFalse(ModuleUrlValidator.isAllowed("http://github.com/egalego-sudo/special-super-intelligency/blob/main/docs/modules/01-model.md"));
        assertFalse(ModuleUrlValidator.isAllowed("https://github.com.evil.example/egalego-sudo/special-super-intelligency/blob/main/docs/modules/01-model.md"));
        assertFalse(ModuleUrlValidator.isAllowed("https://example.com/docs/modules/01-model.md"));
    }

    @Test
    public void rejectsTraversalQueriesAndMalformedPaths() {
        assertFalse(ModuleUrlValidator.isAllowed("https://github.com/egalego-sudo/special-super-intelligency/blob/main/docs/modules/../README.md"));
        assertFalse(ModuleUrlValidator.isAllowed("https://github.com/egalego-sudo/special-super-intelligency/blob/main/docs/modules/01-model.md?redirect=example.com"));
        assertFalse(ModuleUrlValidator.isAllowed("https://github.com/egalego-sudo/special-super-intelligency/blob/main/docs/README.md"));
        assertFalse(ModuleUrlValidator.isAllowed(null));
    }
}
