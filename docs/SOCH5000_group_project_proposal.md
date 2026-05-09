# SOCH 5000 Group Project Proposal

## Responsible Quant Advisory Workbench for Retail Investors
### A Human-in-the-Loop Framework for Quantitative Investing

**Prepared by:** HKUST(GZ) SOCH 5000 Team 1  
**Course:** SOCH 5000 Technological Innovation and Social Entrepreneurship  
**Project Type:** Group project proposal  

---

This proposal presents a retail-oriented investment decision-support framework based on the team's current project. Rather than treating quantitative investing as a purely technical problem, the proposal approaches it as a broader issue involving technology, user protection, and governance. The aim is to show how a machine-learning-based investment system can be made more interpretable, safer, and more accountable for ordinary users.

## 1. Problem Statement

AI and machine learning are increasingly used in quantitative investing, portfolio recommendation, and robo-advisory services. However, in retail-investor settings, these systems often create four serious problems:

First, many AI-driven investing systems remain highly opaque. Retail investors often receive buy or sell recommendations without understanding why the model generated them, what factors drove the decision, or how reliable the recommendation is under current market conditions. This lack of transparency weakens trust and makes it difficult for users to judge whether a recommendation is appropriate for their own risk tolerance and financial goals.

Second, model fragility remains a major concern. Financial markets are highly dynamic, and regime changes can quickly invalidate patterns learned from historical data. A model that performs well in one market environment may become unreliable when volatility rises, correlations shift, or broader macro conditions deteriorate. For retail users, who usually do not have access to institutional-level risk oversight, such model failure can lead to significant losses.

Third, many existing robo-advisory and AI investing products overemphasize return optimization while underemphasizing risk, accountability, and fairness. In practice, retail investors are exposed not only to forecast error, but also to concentration risk, drawdown risk, extreme market shocks, and misleading recommendation framing. Systems that prioritize performance claims without visible guardrails may amplify financial vulnerability rather than reduce it.

Fourth, access to high-quality quantitative tools remains unequal. Advanced quantitative investing infrastructure is typically designed for institutions, while ordinary investors often lack affordable, interpretable, and reviewable tools that combine data-driven recommendations with clear explanations and risk controls. This creates a gap between technological sophistication and socially responsible financial access.

For this reason, the issue is not simply how to build a more accurate model. The more important question is how AI can be used in quantitative investing in a way that improves efficiency while also making investment decision-making more transparent, safer, and more responsible for retail investors.

Our current project is motivated by this gap. The existing system already demonstrates the technical feasibility of an explainable and risk-aware investment workbench through walk-forward modeling, portfolio weighting, volatility and VaR overlays, drawdown controls, and explainability functions. The proposal builds on this technical base and repositions it as a broader solution to a social and governance problem in digital finance.

---

## 2. Project Objective

The objective of this project is to propose a human-centered, AI-assisted quantitative investing framework for retail investors that combines machine-learning-based portfolio recommendation with explainability, risk protection, and user oversight.

More specifically, this project aims to move beyond the conventional logic of treating AI as a black-box return-maximization tool. Instead, it seeks to design a decision-support system that allows retail investors to benefit from data-driven portfolio analysis while still being able to understand, review, and supervise how recommendations are produced.

To achieve this, the project has four main objectives.

First, it aims to improve the quality of portfolio recommendations by using interpretable machine learning methods to extract signals from market data and translate them into structured portfolio suggestions.

Second, it aims to improve transparency by making the recommendation process explainable. Rather than only showing what assets are selected, the system should also communicate why they are selected, which factors drive the recommendation, and what risks are embedded in the portfolio.

Third, it aims to strengthen protection for retail users by incorporating visible risk guardrails, such as drawdown control, volatility thresholds, VaR-aware exposure adjustment, and warning signals under abnormal market conditions.

Fourth, it aims to introduce a governance layer into AI-driven investing by preserving human oversight. In this framework, AI does not replace user judgment entirely; instead, it supports decision-making through interpretable outputs, manual review, and the possibility of intervention or adjustment.

In short, the project objective is not to build an autonomous trading system. Rather, it is to design a more transparent, safer, and more accessible framework through which retail investors can engage with quantitative investing tools.

---

## 3. Proposed Solution

To address the problems identified above, this project proposes the **Responsible Quant Advisory Workbench** for retail investors. The platform is designed as a human-in-the-loop, AI-assisted investment decision-support system rather than a fully automated trading product. Its purpose is not simply to generate portfolio recommendations, but to do so in a way that is interpretable, risk-aware, and governable.

The proposed solution consists of four integrated components.

### 3.1 ML-based signal generation

The system uses interpretable machine learning to evaluate investment opportunities based on:

- price momentum
- volatility and downside risk
- drawdown and regime-related features
- benchmark-relative behavior
- cross-sectional market signals

The signal engine is based on features that are directly relevant to investment decision-making, including momentum, volatility, downside risk, drawdown, moving-average distance, RSI, benchmark-relative beta and correlation, and broader market indicators such as market momentum, market volatility, market drawdown, breadth, and dispersion. This design allows the project to retain a meaningful machine-learning component while remaining understandable for non-expert users.

### 3.2 Portfolio construction and risk-adjusted allocation

The second component translates model signals into portfolio recommendations. Instead of allowing the model to directly dictate unconstrained positions, the system applies a structured portfolio construction process. Positive-return predictions are converted into target weights through a long-only allocation framework, with concentration controlled through top-K asset selection and maximum single-asset weight constraints.

The current project already incorporates several practical risk-adjustment mechanisms at this stage. These include volatility targeting, Value-at-Risk-based exposure scaling, turnover tracking, benchmark-aware backtesting, cash buffering, and drawdown-based exposure reduction. As a result, the recommended portfolio is not simply an output of model ranking; it is a risk-filtered allocation proposal shaped by visible portfolio rules. This is especially important for retail users, since it helps ensure that recommendations remain conservative and reviewable under uncertain market conditions.

### 3.3 Explainability dashboard

The third component is an explainability dashboard that makes model behavior more transparent to users. One of the central weaknesses of many digital investing tools is that they provide a recommendation without revealing how that recommendation was formed. Our project addresses this problem by embedding interpretation directly into the interface.

The dashboard can show global feature importance, local contribution analysis for individual positions, prediction bucket calibration, and one-factor what-if analysis. It also provides broader interpretive views of expected risk-return trade-offs, rolling volatility, drawdown, stress conditions, and market context. This means that users are able not only to see what assets are recommended, but also to understand which factors drove those recommendations, how the model behaved historically, and how sensitive the output may be to changing conditions.

### 3.4 Governance and protection layer

The fourth component is a governance and protection layer that distinguishes this proposal from a conventional quantitative finance system. The current project already includes important protective mechanisms such as drawdown guardrails, volatility thresholds, VaR-based throttling, and risk-event logging. These functions can be extended and reframed as part of a broader governance architecture for retail-facing advisory use.

In the proposed framework, this layer would include user risk profiling, investment suitability checks, warning messages under abnormal market conditions, manual confirmation or override before acting on recommendations, and an audit trail of portfolio suggestions and user responses. This ensures that the system supports decision-making without eliminating human judgment. It also creates a clearer division of responsibility between algorithmic recommendation, platform design, and user action.

Taken together, these four components form a coherent platform logic: machine learning generates signals, portfolio construction transforms them into investable recommendations, explainability makes them understandable, and governance makes them safer and more accountable.

---

## 4. Thematic Areas Involved

This project mainly draws on two thematic areas of the course.

**First, Financial Technology.**  
The project focuses directly on the application of AI and machine learning in quantitative investing and robo-advisory. It uses data-driven models, portfolio construction logic, and risk-control mechanisms to generate and evaluate investment recommendations.

**Second, Innovation and Entrepreneurship.**  
The project is not only a technical model, but also a proposal for a new platform-oriented solution. It presents a concrete product concept that combines machine learning, explainability, and user-centered design to address an unmet need in retail investing.

In addition, the project also incorporates an important governance dimension. Through transparency, risk guardrails, human oversight, and suitability checks, it reflects the course emphasis on making technological innovation more socially responsible and institutionally accountable.

---

## 5. Feasibility and Risks

### 5.1 Feasibility

- **Technical feasibility**  
  The project is technically feasible because it can be built on publicly available financial data and mature machine learning methods. In addition, our current project already provides a working prototype with walk-forward prediction, portfolio construction, risk overlays, and explainability functions, which gives the proposal a strong implementation foundation.

- **Operational feasibility**  
  The proposed system does not require immediate real-market deployment. At the current stage, it can function as a decision-support prototype for simulation, demonstration, and controlled testing. This makes it realistic for a course project while still showing a credible path toward future development.

- **Market feasibility**  
  Retail investors increasingly use digital investment platforms, but many existing systems remain opaque and difficult to supervise. A platform that emphasizes explainability, visible risk controls, and user oversight can offer a meaningful point of differentiation.

- **Course feasibility**  
  The project fits the course requirement well because it is not just a technical exercise. It addresses a real social problem linked to technological change and proposes an actionable innovation that combines finance, entrepreneurship, and governance.

### 5.2 Risks

- **Model risk**  
  Patterns learned from historical market data may not remain valid in future market conditions. A model that performs well in one period may lose predictive power when the market regime changes.

- **Overfitting risk**  
  The system may appear effective in backtesting or sample data but fail to generalize in unseen conditions. This is a common problem in quantitative investing and should be acknowledged explicitly.

- **Behavioral risk**  
  Users may over-rely on machine-generated recommendations and assume the system is more reliable than it actually is. This automation bias can lead to poor decision-making, especially for inexperienced retail investors.

- **Risk of misleading confidence**  
  Even when the model is explainable, users may interpret detailed charts and metrics as proof of certainty. Explainability improves transparency, but it does not remove uncertainty.

- **Governance and accountability risk**  
  If the system gives poor recommendations and users suffer losses, responsibility may become unclear. This creates an important governance challenge concerning disclosure, oversight, and user protection.

### 5.3 Risk Mitigation

- Use interpretable models and explanation tools to reduce opacity and improve user understanding.
- Adopt risk guardrails such as drawdown limits, volatility thresholds, and VaR-based exposure control to reduce downside risk.
- Keep human oversight in the loop by allowing review, adjustment, or rejection of recommendations rather than fully automating decisions.
- Emphasize clear disclosure so that the platform is understood as a decision-support system, not a guaranteed profit engine.
- Use conservative evaluation and stress testing to reduce the risk of overfitting and false confidence.

---

## 6. Potential Societal Impact

This project has potential societal value beyond its technical application in quantitative investing. Its significance lies in how it rethinks the relationship between digital finance, machine learning, and ordinary users.

- **Improving transparency in financial decision-making**  
  The project helps shift investment recommendation systems away from black-box outputs toward more interpretable and explainable processes. This can improve users' understanding of why recommendations are made and what risks they involve.

- **Enhancing protection for retail investors**  
  By integrating risk guardrails, warning mechanisms, and human oversight, the proposed system can reduce the likelihood that retail users blindly follow automated suggestions. This is especially important in volatile or rapidly changing market conditions.

- **Expanding access to data-driven investing tools**  
  High-quality quantitative investing tools are often concentrated in institutional settings. A platform designed for retail accessibility can make more advanced financial analysis available to a broader group of users in a more understandable form.

- **Encouraging more responsible use of AI in finance**  
  The project promotes a model of technology adoption that emphasizes accountability, transparency, and governance rather than pure automation. In this sense, it can serve as an example of how innovation in finance can be aligned with social responsibility.

- **Supporting financial literacy and informed participation**  
  Because the system explains recommendations and highlights risk conditions, it can also function as an educational tool. This may help users develop more informed and critical attitudes toward digital financial products.

Overall, the project's social value lies not in claiming to maximize returns, but in proposing a safer and more transparent way for retail investors to interact with data-driven financial technology.

---

## 7. Team Division of Work

To ensure that the project is both analytically rigorous and presentation-ready, the work can be divided among four members according to the actual structure of this project.

### Member 1: Problem and Background

This member is responsible for framing the project as a SOCH 5000 proposal rather than a purely technical finance task. Main responsibilities include:

- reviewing the current development of AI and machine learning in quantitative investing
- identifying the key problems faced by retail investors
- defining the project as a social and governance issue, not only a technical one
- writing the background, problem statement, and thematic-area relevance

### Member 2: Technical Solution

This member is responsible for explaining the core system design of the project. Main responsibilities include:

- presenting the machine-learning signal generation framework
- explaining data inputs and feature construction
- describing the portfolio construction logic
- summarizing the existing functions already implemented in the current project, such as walk-forward prediction, weighting rules, and risk overlays

### Member 3: Governance and Risk

This member is responsible for the project's protection and accountability dimension. Main responsibilities include:

- explaining the role of explainability in the system
- describing human oversight and manual intervention mechanisms
- discussing risk guardrails such as volatility control, VaR constraints, and drawdown protection
- analyzing broader governance issues including suitability, responsibility, and user protection

### Member 4: Integration and Presentation

This member is responsible for turning the project into a coherent final proposal. Main responsibilities include:

- integrating all sections into a unified proposal narrative
- designing slides and presentation flow
- organizing the written report structure
- refining the overall communication of the project, including diagrams, interface presentation, and storytelling

This division allows the team to cover the project from four complementary angles: social problem definition, technical system design, governance and risk analysis, and final integration and communication. It also matches the interdisciplinary nature of the project and the expectations of the course.
