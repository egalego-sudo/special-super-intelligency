package com.bittupandit.ssi;

import java.util.regex.Pattern;

/** Allows only module documentation paths hosted in this project's GitHub repository. */
final class ModuleUrlValidator {
    private static final Pattern ALLOWED_DOC_URL = Pattern.compile(
        "^https://github[.]com/egalego-sudo/special-super-intelligency/blob/"
            + "(?:android-control-app-v1|main)/docs/modules/[0-9]{2}-[a-z0-9-]+[.]md$"
    );

    private ModuleUrlValidator() { }

    static boolean isAllowed(String url) {
        return url != null && ALLOWED_DOC_URL.matcher(url).matches();
    }
}
